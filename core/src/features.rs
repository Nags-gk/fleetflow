//! Fleet-state tensor used by the learned congestion model (shared by the trainer, the browser and
//! the native benchmark so all three see identical inputs).

use crate::grid::Tile;
use crate::sim::Sim;

pub const CHANNELS: usize = 5;
/// steps of greedy shortest-path roll-out used for the "intent" channel
pub const INTENT_STEPS: usize = 8;
/// label window: mean occupancy over the next `FUTURE` ticks
pub const FUTURE: usize = 8;

/// CHW float tensor: [shelf, station, robots, targets, intent].
pub fn encode(sim: &mut Sim) -> Vec<f32> {
    let (w, h) = (sim.grid.width, sim.grid.height);
    let hw = w * h;
    let mut x = vec![0.0f32; CHANNELS * hw];
    for c in 0..hw {
        match sim.grid.tiles[c] {
            Tile::Shelf => x[c] = 1.0,
            Tile::Station => x[hw + c] = 1.0,
            Tile::Free => {}
        }
    }
    let robots: Vec<(usize, Option<usize>)> =
        sim.robots.iter().map(|r| (r.pos, r.target())).collect();
    for (pos, target) in robots {
        x[2 * hw + pos] = 1.0;
        if let Some(t) = target {
            x[3 * hw + t] = (x[3 * hw + t] + 0.5).min(1.0);
            let d = sim.distances(t);
            let mut cur = pos;
            for _ in 0..INTENT_STEPS {
                if d[cur] == 0 {
                    break;
                }
                let next = sim.grid.neighbors(cur).filter(|&n| d[n] < d[cur]).min();
                match next {
                    Some(n) => {
                        cur = n;
                        x[4 * hw + cur] += 0.25;
                    }
                    None => break,
                }
            }
        }
    }
    for v in &mut x[4 * hw..] {
        *v = v.min(1.0);
    }
    x
}

/// Training label from a window of future position snapshots: per-cell mean occupancy.
pub fn occupancy_label(width: usize, height: usize, future: &[Vec<usize>]) -> Vec<f32> {
    let mut y = vec![0.0f32; width * height];
    for snap in future {
        for &p in snap {
            y[p] += 1.0;
        }
    }
    let n = future.len().max(1) as f32;
    y.iter_mut().for_each(|v| *v /= n);
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{Config, Sim};

    #[test]
    fn encodes_expected_shape_and_ranges() {
        let mut s = Sim::new(Config {
            robots: 10,
            ..Default::default()
        });
        s.run(5);
        let x = encode(&mut s);
        let hw = s.grid.width * s.grid.height;
        assert_eq!(x.len(), CHANNELS * hw);
        assert!(x.iter().all(|v| (0.0..=1.0).contains(v)));
        let robots: f32 = x[2 * hw..3 * hw].iter().sum();
        assert_eq!(robots as usize, 10);
    }
}
