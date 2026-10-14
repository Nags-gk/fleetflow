//! Lifelong multi-robot warehouse simulator.
//!
//! Robots continuously receive pick tasks (shelf pod -> pick station). Every `replan_every` ticks,
//! or as soon as a robot finishes a leg (event-triggered), the fleet manager:
//! 1. assigns open tasks to idle robots (Hungarian or greedy),
//! 2. re-plans every robot with windowed prioritized space-time A* (Rolling-Horizon Collision
//!    Resolution), optionally with a congestion cost layer,
//! 3. executes the first steps of each plan.
//!
//! An independent validator checks every tick for vertex and swap collisions.

use crate::assign;
use crate::astar::{plan, Query, Reservations, UNIT};
use crate::grid::{Cell, DistCache, Grid};
use crate::rng::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Assigner {
    Hungarian,
    Greedy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CongestionSource {
    /// plain shortest-time planning
    None,
    /// reactive: penalise cells near robots right now (blurred occupancy)
    Density,
    /// predictive: a cost layer supplied from outside (the learned model)
    External,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub width: usize,
    pub height: usize,
    pub block_len: usize,
    pub robots: usize,
    pub seed: u64,
    pub horizon: u32,
    pub replan_every: u32,
    pub assigner: Assigner,
    pub congestion: CongestionSource,
    pub congestion_weight: f32,
    /// only congestion above this level is penalised (hotspot avoidance, not a traffic tax)
    pub congestion_threshold: f32,
    /// open tasks kept in the pool, as a multiple of fleet size
    pub task_pool: f32,
    /// choose the drop-off station at pickup time by distance + queue length (vs. fixed random)
    pub balance_stations: bool,
    /// extra distance-equivalent cost per robot already queued for a station
    pub station_queue_penalty: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            width: 40,
            height: 26,
            block_len: 5,
            robots: 40,
            seed: 1,
            horizon: 10,
            replan_every: 2,
            assigner: Assigner::Hungarian,
            congestion: CongestionSource::None,
            congestion_weight: 0.0,
            congestion_threshold: 0.15,
            task_pool: 1.5,
            balance_stations: true,
            station_queue_penalty: 6,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Phase {
    ToPickup,
    ToDrop,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Task {
    pub pickup: Cell,
    pub drop: Cell,
    pub created: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Robot {
    pub pos: Cell,
    pub task: Option<(Task, Phase, u32)>, // (task, phase, assigned_at)
    #[serde(skip)]
    pub plan: VecDeque<Cell>,
    pub waits: u32,
}

impl Robot {
    pub fn target(&self) -> Option<Cell> {
        self.task.map(|(t, ph, _)| {
            if ph == Phase::ToPickup {
                t.pickup
            } else {
                t.drop
            }
        })
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Metrics {
    pub tick: u32,
    pub tasks_completed: u32,
    /// completed tasks per 1000 ticks
    pub throughput: f32,
    /// mean ticks from assignment to drop-off
    pub mean_task_time: f32,
    /// mean ticks from task creation to drop-off (includes queueing)
    pub mean_service_time: f32,
    pub total_moves: u64,
    pub total_waits: u64,
    pub collisions: u32,
    pub replans: u32,
    pub fallback_stops: u32,
    /// replans that needed the guaranteed-feasible conservative pass
    pub conservative_replans: u32,
    pub low_level_expansions: u64,
    pub assignment_cost: i64,
    pub plan_ms_total: f64,
}

pub struct Sim {
    pub cfg: Config,
    pub grid: Grid,
    pub robots: Vec<Robot>,
    pub open_tasks: Vec<Task>,
    pub tick: u32,
    pub metrics: Metrics,
    /// external / computed congestion map in [0, 1] per cell
    pub congestion: Vec<f32>,
    rng: Rng,
    dc: DistCache,
    access: Vec<Cell>,
    stations: Vec<Cell>,
    event: bool,
    sum_task_time: u64,
    sum_service_time: u64,
    /// rolling history of robot positions (for training targets)
    pub history: VecDeque<Vec<Cell>>,
    pub now_ms: fn() -> f64,
}

fn zero_ms() -> f64 {
    0.0
}

impl Sim {
    pub fn new(cfg: Config) -> Self {
        let grid = Grid::warehouse(cfg.width, cfg.height, cfg.block_len);
        Self::with_grid(cfg, grid)
    }

    pub fn with_grid(cfg: Config, grid: Grid) -> Self {
        let mut rng = Rng::new(cfg.seed);
        let mut free = grid.free_cells();
        rng.shuffle(&mut free);
        let n = cfg.robots.min(free.len());
        let robots = free[..n]
            .iter()
            .map(|&pos| Robot {
                pos,
                task: None,
                plan: VecDeque::new(),
                waits: 0,
            })
            .collect();
        let access = grid.shelf_access_points();
        let stations = grid.stations();
        let cells = grid.tiles.len();
        let mut s = Sim {
            cfg,
            grid,
            robots,
            open_tasks: vec![],
            tick: 0,
            metrics: Metrics::default(),
            congestion: vec![0.0; cells],
            rng,
            dc: DistCache::default(),
            access,
            stations,
            event: true,
            sum_task_time: 0,
            sum_service_time: 0,
            history: VecDeque::new(),
            now_ms: zero_ms,
        };
        s.refill_tasks();
        s
    }

    fn refill_tasks(&mut self) {
        let want = ((self.robots.len() as f32) * self.cfg.task_pool).ceil() as usize;
        while self.open_tasks.len() < want {
            let pickup = self.access[self.rng.below(self.access.len())];
            let drop = self.stations[self.rng.below(self.stations.len())];
            self.open_tasks.push(Task {
                pickup,
                drop,
                created: self.tick,
            });
        }
    }

    fn assign_tasks(&mut self) {
        let idle: Vec<usize> = (0..self.robots.len())
            .filter(|&i| self.robots[i].task.is_none())
            .collect();
        if idle.is_empty() || self.open_tasks.is_empty() {
            return;
        }
        let tasks: Vec<Task> = self.open_tasks.clone();
        // costs: travel distance robot -> pickup (BFS from each pickup)
        let mut cost = Vec::with_capacity(idle.len());
        for &i in &idle {
            let mut row = Vec::with_capacity(tasks.len());
            for t in &tasks {
                let d = self.dc.get(&self.grid, t.pickup);
                row.push(d[self.robots[i].pos] as i64);
            }
            cost.push(row);
        }
        // if more robots than tasks, solve the transposed problem
        let pairs: Vec<(usize, usize)> = if idle.len() <= tasks.len() {
            let a = match self.cfg.assigner {
                Assigner::Hungarian => assign::hungarian(&cost),
                Assigner::Greedy => assign::greedy(&cost),
            };
            a.into_iter()
                .enumerate()
                .filter(|(_, j)| *j != usize::MAX)
                .collect()
        } else {
            let t: Vec<Vec<i64>> = (0..tasks.len())
                .map(|j| cost.iter().map(|r| r[j]).collect())
                .collect();
            let a = match self.cfg.assigner {
                Assigner::Hungarian => assign::hungarian(&t),
                Assigner::Greedy => assign::greedy(&t),
            };
            a.into_iter()
                .enumerate()
                .filter(|(_, i)| *i != usize::MAX)
                .map(|(j, i)| (i, j))
                .collect()
        };
        let mut used = vec![false; tasks.len()];
        for (ri, tj) in pairs {
            self.metrics.assignment_cost += cost[ri][tj];
            let r = idle[ri];
            self.robots[r].task = Some((tasks[tj], Phase::ToPickup, self.tick));
            used[tj] = true;
        }
        self.open_tasks = tasks
            .into_iter()
            .zip(used)
            .filter(|(_, u)| !u)
            .map(|(t, _)| t)
            .collect();
        self.refill_tasks();
    }

    fn update_density_congestion(&mut self) {
        let w = self.grid.width;
        let h = self.grid.height;
        let mut occ = vec![0.0f32; w * h];
        for r in &self.robots {
            occ[r.pos] += 1.0;
        }
        // 5x5 box blur
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0;
                for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                        if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                            s += occ[yy as usize * w + xx as usize];
                        }
                    }
                }
                self.congestion[y * w + x] = (s / 6.0).min(1.0);
            }
        }
    }

    fn scale_penalty(&self, v: f32) -> u32 {
        let tau = self.cfg.congestion_threshold;
        let excess = ((v - tau) / (1.0 - tau)).clamp(0.0, 1.0);
        (excess * self.cfg.congestion_weight * UNIT as f32) as u32
    }

    fn penalty(&self) -> Option<Vec<u32>> {
        if self.cfg.congestion == CongestionSource::None || self.cfg.congestion_weight <= 0.0 {
            return None;
        }
        Some(
            self.congestion
                .iter()
                .map(|&c| self.scale_penalty(c))
                .collect(),
        )
    }

    /// A robot's own expected contribution to the congestion map, removed before it plans so it
    /// is not pushed off its own shortest path ("self-avoidance" bias).
    fn self_footprint(&mut self, i: usize) -> Vec<(Cell, f32)> {
        let pos = self.robots[i].pos;
        match self.cfg.congestion {
            CongestionSource::Density => {
                let (x, y) = self.grid.xy(pos);
                let mut v = vec![];
                for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                        if xx >= 0
                            && yy >= 0
                            && (xx as usize) < self.grid.width
                            && (yy as usize) < self.grid.height
                        {
                            v.push((self.grid.cell(xx as usize, yy as usize), 1.0 / 6.0));
                        }
                    }
                }
                v
            }
            // the learned map predicts *waiting* robots; only the robot's own cell is self-induced
            CongestionSource::External => vec![(pos, 1.0 / crate::features::FUTURE as f32)],
            CongestionSource::None => vec![],
        }
    }

    /// Windowed prioritized planning with priority bumping on failure.
    fn replan(&mut self) {
        let t0 = (self.now_ms)();
        self.metrics.replans += 1;
        if self.cfg.congestion == CongestionSource::Density {
            self.update_density_congestion();
        }
        let penalty = self.penalty();
        let n = self.robots.len();
        // priority: loaded robots first, then by remaining distance (longer first)
        let mut order: Vec<usize> = (0..n).collect();
        let key = |s: &mut Sim, i: usize| -> (u8, u32) {
            let r = &s.robots[i];
            match r.task {
                Some((_, Phase::ToDrop, _)) => {
                    (0, u32::MAX - s.dc.get(&s.grid, r.target().unwrap())[r.pos])
                }
                Some(_) => (1, u32::MAX - s.dc.get(&s.grid, r.target().unwrap())[r.pos]),
                None => (2, 0),
            }
        };
        let mut keys: Vec<(u8, u32)> = (0..n).map(|i| key(self, i)).collect();
        // rotate ties by tick to avoid starvation
        order.sort_by_key(|&i| (keys[i], (i + self.tick as usize) % n));
        keys.clear();

        let horizon = self.cfg.horizon;
        // Attempts 0..=5 are "optimistic": a robot not yet planned only blocks its cell for t <= 1,
        // so higher-priority robots may route through cells it is expected to vacate. If a robot
        // gets boxed in, retry with it at top priority, then with random priority restarts.
        // The final attempt is "conservative": unplanned robots block their cell for the whole
        // window, which guarantees every robot can at least wait in place, so it always succeeds.
        const OPTIMISTIC_ATTEMPTS: usize = 6;
        for attempt in 0..=OPTIMISTIC_ATTEMPTS {
            let conservative = attempt == OPTIMISTIC_ATTEMPTS;
            let mut res = Reservations::new();
            let mut plans: Vec<Option<Vec<Cell>>> = vec![None; n];
            let mut failed = None;
            for (k, &i) in order.iter().enumerate() {
                let mut local = res.clone();
                let hold = if conservative { horizon } else { 1 };
                for &j in &order[k + 1..] {
                    for t in 0..=hold {
                        local.reserve_vertex(self.robots[j].pos, t);
                    }
                }
                let overrides: Option<std::collections::HashMap<Cell, u32>> =
                    penalty.as_ref().map(|_| {
                        let mut m = std::collections::HashMap::new();
                        for (c, d) in self.self_footprint(i) {
                            let e = m.entry(c).or_insert(self.congestion[c]);
                            *e -= d;
                        }
                        m.into_iter()
                            .map(|(c, v)| (c, self.scale_penalty(v)))
                            .collect()
                    });
                let r = &self.robots[i];
                let goal = r.target().unwrap_or(r.pos);
                let h = self.dc.get(&self.grid, goal);
                let q = Query {
                    penalty_override: overrides.as_ref(),
                    grid: &self.grid,
                    start: r.pos,
                    goal,
                    h: &h,
                    res: &local,
                    horizon: Some(horizon),
                    penalty: penalty.as_deref(),
                    max_expansions: 50_000,
                };
                match plan(&q) {
                    Some(p) => {
                        self.metrics.low_level_expansions += p.expansions as u64;
                        res.reserve_path(&p.path, p.reached_goal);
                        plans[i] = Some(p.path);
                    }
                    None => {
                        failed = Some(i);
                        break;
                    }
                }
            }
            match failed {
                None => {
                    for (i, p) in plans.into_iter().enumerate() {
                        let p = p.unwrap();
                        let steps = self.cfg.replan_every as usize;
                        let mut q: VecDeque<Cell> = p.iter().skip(1).take(steps).copied().collect();
                        let last = *p.last().unwrap();
                        while q.len() < steps {
                            q.push_back(last);
                        }
                        self.robots[i].plan = q;
                    }
                    if conservative {
                        self.metrics.conservative_replans += 1;
                    }
                    self.metrics.plan_ms_total += (self.now_ms)() - t0;
                    return;
                }
                Some(i) => {
                    if attempt < 2 {
                        order.retain(|&x| x != i);
                        order.insert(0, i);
                    } else {
                        self.rng.shuffle(&mut order);
                    }
                }
            }
        }
        // safe fallback: everyone holds position until next replan
        self.metrics.fallback_stops += 1;
        for r in &mut self.robots {
            r.plan = std::iter::repeat_n(r.pos, self.cfg.replan_every as usize).collect();
        }
        self.metrics.plan_ms_total += (self.now_ms)() - t0;
    }

    /// True if the next `step()` will re-plan (callers refresh an external congestion map first).
    pub fn needs_replan(&self) -> bool {
        self.event
            || self.tick.is_multiple_of(self.cfg.replan_every)
            || self.robots.iter().any(|r| r.plan.is_empty())
    }

    pub fn step(&mut self) {
        if self.needs_replan() {
            self.assign_tasks();
            self.replan();
            self.event = false;
        }
        let prev: Vec<Cell> = self.robots.iter().map(|r| r.pos).collect();
        for r in self.robots.iter_mut() {
            let next = r.plan.pop_front().unwrap_or(r.pos);
            if next == r.pos {
                if r.task.is_some() {
                    r.waits += 1;
                    self.metrics.total_waits += 1;
                }
            } else {
                self.metrics.total_moves += 1;
            }
            r.pos = next;
        }
        self.tick += 1;
        self.metrics.collisions += count_collisions(
            &prev,
            &self.robots.iter().map(|r| r.pos).collect::<Vec<_>>(),
        );

        // task progression
        for i in 0..self.robots.len() {
            let (pos, cur, target) = (
                self.robots[i].pos,
                self.robots[i].task,
                self.robots[i].target(),
            );
            if let Some((mut task, phase, assigned)) = cur {
                if Some(pos) == target {
                    self.event = true;
                    match phase {
                        Phase::ToPickup => {
                            if self.cfg.balance_stations {
                                task.drop = self.pick_station(pos);
                            }
                            self.robots[i].task = Some((task, Phase::ToDrop, assigned));
                        }
                        Phase::ToDrop => {
                            self.robots[i].task = None;
                            self.metrics.tasks_completed += 1;
                            self.sum_task_time += (self.tick - assigned) as u64;
                            self.sum_service_time += (self.tick - task.created) as u64;
                        }
                    }
                }
            }
        }
        self.history
            .push_back(self.robots.iter().map(|r| r.pos).collect());
        if self.history.len() > 64 {
            self.history.pop_front();
        }
        let m = &mut self.metrics;
        m.tick = self.tick;
        m.throughput = 1000.0 * m.tasks_completed as f32 / self.tick.max(1) as f32;
        if m.tasks_completed > 0 {
            m.mean_task_time = self.sum_task_time as f32 / m.tasks_completed as f32;
            m.mean_service_time = self.sum_service_time as f32 / m.tasks_completed as f32;
        }
    }

    /// Load-balanced station choice: travel distance + queue penalty per robot already headed there.
    fn pick_station(&mut self, from: Cell) -> Cell {
        let mut load = std::collections::HashMap::new();
        for r in &self.robots {
            if let Some((t, Phase::ToDrop, _)) = r.task {
                *load.entry(t.drop).or_insert(0u32) += 1;
            }
        }
        let stations = self.stations.clone();
        let mut best = (u32::MAX, stations[0]);
        for st in stations {
            let d = self.dc.get(&self.grid, st)[from];
            let score = d.saturating_add(
                self.cfg.station_queue_penalty * load.get(&st).copied().unwrap_or(0),
            );
            if score < best.0 {
                best = (score, st);
            }
        }
        best.1
    }

    pub fn run(&mut self, ticks: u32) {
        for _ in 0..ticks {
            self.step();
        }
    }

    pub fn reset_metrics(&mut self) {
        self.metrics = Metrics::default();
        self.sum_task_time = 0;
        self.sum_service_time = 0;
    }

    pub fn distances(&mut self, goal: Cell) -> std::rc::Rc<Vec<u32>> {
        self.dc.get(&self.grid, goal)
    }
}

/// Independent collision validator: vertex conflicts and head-on swaps between two snapshots.
pub fn count_collisions(prev: &[Cell], next: &[Cell]) -> u32 {
    let mut c = 0;
    let mut seen = std::collections::HashMap::new();
    for (i, &p) in next.iter().enumerate() {
        if let Some(_j) = seen.insert(p, i) {
            c += 1;
        }
    }
    let mut moves = std::collections::HashSet::new();
    for i in 0..prev.len() {
        if prev[i] != next[i] {
            moves.insert((prev[i], next[i]));
        }
    }
    for &(a, b) in &moves {
        if a < b && moves.contains(&(b, a)) {
            c += 1;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifelong_run_is_collision_free_and_productive() {
        for seed in 0..3 {
            let mut s = Sim::new(Config {
                robots: 30,
                seed,
                ..Default::default()
            });
            s.run(400);
            assert_eq!(s.metrics.collisions, 0, "seed {seed}");
            assert!(
                s.metrics.tasks_completed > 20,
                "seed {seed}: {:?}",
                s.metrics
            );
        }
    }

    #[test]
    fn deterministic_replay() {
        let run = || {
            let mut s = Sim::new(Config {
                robots: 20,
                seed: 9,
                ..Default::default()
            });
            s.run(200);
            (
                s.metrics.tasks_completed,
                s.robots.iter().map(|r| r.pos).collect::<Vec<_>>(),
            )
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn hungarian_assignment_cost_not_worse_than_greedy_on_first_round() {
        let mk = |a| {
            let mut s = Sim::new(Config {
                robots: 25,
                seed: 3,
                assigner: a,
                ..Default::default()
            });
            s.step();
            s.metrics.assignment_cost
        };
        assert!(mk(Assigner::Hungarian) <= mk(Assigner::Greedy));
    }

    #[test]
    fn validator_flags_swaps_and_overlaps() {
        assert_eq!(count_collisions(&[0, 1], &[1, 0]), 1);
        assert_eq!(count_collisions(&[0, 2], &[1, 1]), 1);
        assert_eq!(count_collisions(&[0, 2], &[1, 3]), 0);
    }
}
