//! End-to-end: the trained congestion CNN drives the planner's cost layer.

use fleetflow_core::cnn::Cnn;
use fleetflow_core::features;
use fleetflow_core::sim::{Config, CongestionSource, Sim};

const MODEL: &str = include_str!("../../model/congestion.json");

fn run(robots: usize, learned: bool, ticks: u32) -> Sim {
    let model = Cnn::from_json(MODEL).unwrap();
    let cfg = Config {
        robots,
        seed: 4242,
        congestion: if learned {
            CongestionSource::External
        } else {
            CongestionSource::None
        },
        congestion_weight: 3.0,
        ..Default::default()
    };
    let mut s = Sim::new(cfg);
    for _ in 0..ticks {
        if learned && s.needs_replan() {
            let x = features::encode(&mut s);
            s.congestion = model.forward(&x, s.grid.height, s.grid.width);
        }
        s.step();
    }
    s
}

#[test]
fn model_output_is_a_valid_cost_map() {
    let model = Cnn::from_json(MODEL).unwrap();
    let mut s = Sim::new(Config {
        robots: 50,
        ..Default::default()
    });
    s.run(30);
    let y = model.forward(&features::encode(&mut s), s.grid.height, s.grid.width);
    assert_eq!(y.len(), s.grid.width * s.grid.height);
    assert!(y.iter().all(|v| (0.0..=1.0).contains(v)));
}

#[test]
fn learned_planner_is_collision_free_and_avoids_gridlock() {
    let base = run(110, false, 500);
    let learned = run(110, true, 500);
    assert_eq!(learned.metrics.collisions, 0);
    assert_eq!(base.metrics.collisions, 0);
    assert!(
        learned.metrics.tasks_completed > base.metrics.tasks_completed,
        "learned {} vs baseline {}",
        learned.metrics.tasks_completed,
        base.metrics.tasks_completed
    );
}
