//! fleetflow CLI: benchmarks and training-data generation.
//!
//!   ffsim bench   [--ticks N] [--seeds K] [--model model.json]      > bench.json
//!   ffsim mapf    [--instances N]                                   > mapf.json
//!   ffsim dataset --out data.bin [--samples N]

use fleetflow_core::cbs::{self, Agent};
use fleetflow_core::cnn::Cnn;
use fleetflow_core::features::{self, FUTURE};
use fleetflow_core::grid::Grid;
use fleetflow_core::rng::Rng;
use fleetflow_core::sim::{Assigner, Config, CongestionSource, Sim};
use serde_json::json;
use std::io::Write;
use std::time::Instant;

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn now_ms() -> f64 {
    thread_local!(static T0: Instant = Instant::now());
    T0.with(|t| t.elapsed().as_secs_f64() * 1000.0)
}

/// Run one simulation, refreshing the congestion map from `model` before every replan.
fn run(cfg: Config, ticks: u32, warmup: u32, model: Option<&Cnn>) -> serde_json::Value {
    let mut s = Sim::new(cfg);
    s.now_ms = now_ms;
    let (h, w) = (s.grid.height, s.grid.width);
    let mut infer_ms = 0.0;
    for t in 0..(warmup + ticks) {
        if t == warmup {
            s.reset_metrics();
        }
        if let (Some(m), true) = (model, s.needs_replan()) {
            let t0 = now_ms();
            let x = features::encode(&mut s);
            s.congestion = m.forward(&x, h, w);
            infer_ms += now_ms() - t0;
        }
        s.step();
    }
    let m = &s.metrics;
    json!({
        "throughput": m.throughput, "mean_task_time": m.mean_task_time, "tasks": m.tasks_completed,
        "waits": m.total_waits, "moves": m.total_moves, "collisions": m.collisions,
        "fallback_stops": m.fallback_stops, "plan_ms_per_replan": m.plan_ms_total / m.replans.max(1) as f64,
        "infer_ms_per_replan": infer_ms / m.replans.max(1) as f64,
    })
}

fn bench(args: &[String]) {
    let ticks = arg(args, "--ticks", 1500u32);
    let seeds = arg(args, "--seeds", 5u64);
    let model = args.iter().position(|a| a == "--model").map(|i| {
        Cnn::from_json(&std::fs::read_to_string(&args[i + 1]).expect("model file"))
            .expect("model json")
    });
    let weight = arg(args, "--weight", 3.0f32);
    let mut rows = vec![];
    for &robots in &[20usize, 40, 60, 80, 100, 120] {
        let mut variants: Vec<(&str, Assigner, CongestionSource, f32)> = vec![
            ("greedy", Assigner::Greedy, CongestionSource::None, 0.0),
            (
                "hungarian",
                Assigner::Hungarian,
                CongestionSource::None,
                0.0,
            ),
            (
                "hungarian+density",
                Assigner::Hungarian,
                CongestionSource::Density,
                weight,
            ),
        ];
        if model.is_some() {
            variants.push((
                "hungarian+learned",
                Assigner::Hungarian,
                CongestionSource::External,
                weight,
            ));
        }
        for (name, a, c, wgt) in variants {
            let mut runs = vec![];
            for seed in 0..seeds {
                let cfg = Config {
                    robots,
                    seed: 1000 + seed,
                    assigner: a,
                    congestion: c,
                    congestion_weight: wgt,
                    ..Default::default()
                };
                let m = if c == CongestionSource::External {
                    model.as_ref()
                } else {
                    None
                };
                runs.push(run(cfg, ticks, 200, m));
            }
            let mean = |k: &str| {
                runs.iter().map(|r| r[k].as_f64().unwrap()).sum::<f64>() / runs.len() as f64
            };
            let row = json!({
                "robots": robots, "variant": name, "seeds": seeds, "ticks": ticks,
                "throughput": mean("throughput"), "mean_task_time": mean("mean_task_time"),
                "waits": mean("waits"), "collisions": mean("collisions"), "fallback_stops": mean("fallback_stops"),
                "plan_ms_per_replan": mean("plan_ms_per_replan"), "infer_ms_per_replan": mean("infer_ms_per_replan"),
                "runs": runs,
            });
            eprintln!("{robots:>3} robots {name:<20} thr {:.1}/1k ticks  task {:.1}  waits {:.0}  coll {}",
                row["throughput"].as_f64().unwrap(), row["mean_task_time"].as_f64().unwrap(), row["waits"].as_f64().unwrap(), row["collisions"]);
            rows.push(row);
        }
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
}

fn mapf(args: &[String]) {
    let n_inst = arg(args, "--instances", 25usize);
    let grid = Grid::warehouse(24, 13, 4);
    let free = grid.free_cells();
    let mut rng = Rng::new(2024);
    let mut rows = vec![];
    for &k in &[4usize, 8, 12, 16] {
        let (mut c_ok, mut p_ok, mut both, mut gap_sum, mut c_ms, mut p_ms) =
            (0, 0, 0, 0.0, 0.0, 0.0);
        let mut ct_nodes = 0usize;
        for _ in 0..n_inst {
            let mut cells = free.clone();
            rng.shuffle(&mut cells);
            let agents: Vec<Agent> = (0..k)
                .map(|i| Agent {
                    start: cells[i],
                    goal: cells[i + k],
                })
                .collect();
            let t0 = Instant::now();
            let c = cbs::cbs(&grid, &agents, 20_000);
            c_ms += t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let p = cbs::prioritized(&grid, &agents);
            p_ms += t1.elapsed().as_secs_f64() * 1e3;
            if let Some(c) = &c {
                c_ok += 1;
                ct_nodes += c.high_level_nodes;
                assert!(cbs::first_conflict(&c.paths).is_none());
            }
            if p.is_some() {
                p_ok += 1;
            }
            if let (Some(c), Some(p)) = (&c, &p) {
                both += 1;
                gap_sum += (p.sum_of_costs as f64 - c.sum_of_costs as f64) / c.sum_of_costs as f64;
            }
        }
        let row = json!({
            "agents": k, "instances": n_inst,
            "cbs_success": c_ok as f64 / n_inst as f64, "prioritized_success": p_ok as f64 / n_inst as f64,
            "prioritized_soc_gap_pct": if both > 0 { 100.0 * gap_sum / both as f64 } else { 0.0 },
            "cbs_ms_mean": c_ms / n_inst as f64, "prioritized_ms_mean": p_ms / n_inst as f64,
            "cbs_ct_nodes_mean": if c_ok > 0 { ct_nodes as f64 / c_ok as f64 } else { 0.0 },
        });
        eprintln!("{row}");
        rows.push(row);
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
}

/// Binary dataset: header line (JSON) then f32 little-endian [x (C*H*W), y (H*W)] per sample.
fn dataset(args: &[String]) {
    let out = arg(args, "--out", String::from("data.bin"));
    let samples = arg(args, "--samples", 4000usize);
    let waits_label = arg(args, "--label", String::from("waits")) == "waits";
    let mut rng = Rng::new(77);
    let base = Config::default();
    let (h, w) = (base.height, base.width);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    let header = json!({ "c": features::CHANNELS, "h": h, "w": w, "future": FUTURE, "samples": samples, "label": if waits_label { "waits" } else { "occupancy" } });
    writeln!(f, "{header}").unwrap();
    let mut written = 0;
    let mut sim_id = 0u64;
    while written < samples {
        let robots = 20 + rng.below(100);
        let congestion = if rng.below(2) == 0 {
            CongestionSource::None
        } else {
            CongestionSource::Density
        };
        let cfg = Config {
            robots,
            seed: 50_000 + sim_id,
            congestion,
            congestion_weight: 2.0,
            ..base.clone()
        };
        sim_id += 1;
        let mut s = Sim::new(cfg);
        s.run(60);
        for _ in 0..40 {
            if written >= samples {
                break;
            }
            let x = features::encode(&mut s);
            let mut fut = vec![];
            for _ in 0..FUTURE {
                let before: Vec<(usize, bool)> =
                    s.robots.iter().map(|r| (r.pos, r.task.is_some())).collect();
                s.step();
                fut.push(if waits_label {
                    // congestion = robots that had work to do but could not move
                    s.robots
                        .iter()
                        .zip(&before)
                        .filter(|(r, (p, busy))| *busy && r.pos == *p)
                        .map(|(r, _)| r.pos)
                        .collect()
                } else {
                    s.robots.iter().map(|r| r.pos).collect::<Vec<_>>()
                });
            }
            let y = features::occupancy_label(w, h, &fut);
            for v in x.iter().chain(y.iter()) {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            written += 1;
            // decorrelate consecutive samples
            s.run(1 + rng.below(6) as u32);
        }
    }
    eprintln!("wrote {written} samples to {out} from {sim_id} simulations");
}

/// Check native Rust inference against ONNX Runtime outputs saved by the trainer.
fn parity(args: &[String]) {
    let model = Cnn::from_json(
        &std::fs::read_to_string(arg(args, "--model", String::from("congestion.json"))).unwrap(),
    )
    .unwrap();
    let fx: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(arg(args, "--fixture", String::from("parity_fixture.json")))
            .unwrap(),
    )
    .unwrap();
    let (h, w) = (
        fx["h"].as_u64().unwrap() as usize,
        fx["w"].as_u64().unwrap() as usize,
    );
    let tol = arg(args, "--tol", 1e-4f32);
    let mut worst = 0.0f32;
    for (x, y) in fx["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .zip(fx["outputs"].as_array().unwrap())
    {
        let x: Vec<f32> = serde_json::from_value(x.clone()).unwrap();
        let y: Vec<f32> = serde_json::from_value(y.clone()).unwrap();
        let p = model.forward(&x, h, w);
        worst = p
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(worst, f32::max);
    }
    println!(
        "{}",
        json!({ "max_abs_diff": worst, "tolerance": tol, "ok": worst <= tol })
    );
    if worst > tol {
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("bench") => bench(&args),
        Some("mapf") => mapf(&args),
        Some("dataset") => dataset(&args),
        Some("parity") => parity(&args),
        _ => eprintln!("usage: ffsim <bench|mapf|dataset> [options]"),
    }
}
