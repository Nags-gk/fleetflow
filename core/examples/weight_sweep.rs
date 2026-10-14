use fleetflow_core::{cnn::Cnn, features, sim::*};
fn run(robots: usize, seed: u64, c: CongestionSource, w: f32, m: Option<&Cnn>) -> f32 {
    let mut s = Sim::new(Config {
        robots,
        seed,
        congestion: c,
        congestion_weight: w,
        ..Default::default()
    });
    for t in 0..800 {
        if t == 200 {
            s.reset_metrics();
        }
        if let (Some(m), true) = (m, s.needs_replan()) {
            let x = features::encode(&mut s);
            s.congestion = m.forward(&x, s.grid.height, s.grid.width);
        }
        s.step();
    }
    s.metrics.throughput
}
fn main() {
    let m = Cnn::from_json(&std::fs::read_to_string("../model/congestion.json").unwrap()).unwrap();
    for robots in [40usize, 80, 110] {
        let base: f32 = (0..2)
            .map(|sd| run(robots, 900 + sd, CongestionSource::None, 0.0, None))
            .sum::<f32>()
            / 2.0;
        print!("{robots}: none {base:.0} |");
        for w in [1.0f32, 3.0, 6.0] {
            let d: f32 = (0..2)
                .map(|sd| run(robots, 900 + sd, CongestionSource::Density, w, None))
                .sum::<f32>()
                / 2.0;
            let l: f32 = (0..2)
                .map(|sd| run(robots, 900 + sd, CongestionSource::External, w, Some(&m)))
                .sum::<f32>()
                / 2.0;
            print!(" w{w}: dens {d:.0} learned {l:.0} |");
        }
        println!();
    }
}
