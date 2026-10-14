//! Space-time A* over (cell, timestep) with a reservation table.
//!
//! Used two ways:
//! * windowed: prioritized planning for the lifelong simulator (plans `horizon` steps ahead),
//! * full: low-level search inside Conflict-Based Search (plans until the agent can rest at its goal).

use crate::grid::{Cell, Grid};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Integer cost scale: one unit move = `UNIT`. Congestion penalties are added on top.
pub const UNIT: u32 = 100;

/// Vertex / edge occupancy in space-time. Doubles as the CBS constraint set.
#[derive(Clone, Default, Debug)]
pub struct Reservations {
    vertex: HashSet<(Cell, u32)>,
    edge: HashSet<(Cell, Cell, u32)>, // move from -> to, arriving at t
    /// directed CBS edge constraints: this agent may not move from -> to arriving at t
    forbidden: HashSet<(Cell, Cell, u32)>,
    /// cell is blocked for every t >= value (a robot parked at its goal)
    permanent: HashMap<Cell, u32>,
    /// latest timestep any vertex reservation touches a cell
    last_t: HashMap<Cell, u32>,
}

impl Reservations {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reserve_vertex(&mut self, c: Cell, t: u32) {
        self.vertex.insert((c, t));
        let e = self.last_t.entry(c).or_insert(t);
        if *e < t {
            *e = t;
        }
    }

    pub fn reserve_edge(&mut self, from: Cell, to: Cell, t: u32) {
        self.edge.insert((from, to, t));
    }

    pub fn forbid_move(&mut self, from: Cell, to: Cell, t: u32) {
        self.forbidden.insert((from, to, t));
    }

    pub fn reserve_from(&mut self, c: Cell, t: u32) {
        let e = self.permanent.entry(c).or_insert(t);
        if *e > t {
            *e = t;
        }
    }

    /// Reserve a whole path: every vertex, every traversed edge, and (optionally) the final cell
    /// from arrival onwards.
    pub fn reserve_path(&mut self, path: &[Cell], park_at_end: bool) {
        for (t, &c) in path.iter().enumerate() {
            self.reserve_vertex(c, t as u32);
            if t > 0 && path[t - 1] != c {
                self.reserve_edge(path[t - 1], c, t as u32);
            }
        }
        if park_at_end {
            if let Some(&last) = path.last() {
                self.reserve_from(last, path.len() as u32 - 1);
            }
        }
    }

    #[inline]
    pub fn vertex_blocked(&self, c: Cell, t: u32) -> bool {
        self.vertex.contains(&(c, t)) || self.permanent.get(&c).is_some_and(|&p| t >= p)
    }

    /// Moving from->to arriving at t collides head-on with someone moving to->from.
    #[inline]
    pub fn edge_blocked(&self, from: Cell, to: Cell, t: u32) -> bool {
        self.edge.contains(&(to, from, t))
            || self.edge.contains(&(from, to, t))
            || self.forbidden.contains(&(from, to, t))
    }

    /// Can a robot that arrives at `c` at time `t` stay there forever?
    pub fn can_rest(&self, c: Cell, t: u32) -> bool {
        !self.permanent.contains_key(&c) && self.last_t.get(&c).is_none_or(|&lt| lt < t)
    }

    pub fn max_t(&self) -> u32 {
        self.last_t.values().copied().max().unwrap_or(0)
    }
}

pub struct Query<'a> {
    pub grid: &'a Grid,
    pub start: Cell,
    pub goal: Cell,
    /// exact distance-to-goal table (BFS from goal)
    pub h: &'a [u32],
    pub res: &'a Reservations,
    /// `Some(T)`: windowed search, any node at t == T is terminal.
    pub horizon: Option<u32>,
    /// optional per-cell extra cost (already scaled to UNIT), e.g. predicted congestion
    pub penalty: Option<&'a [u32]>,
    /// per-agent overrides of `penalty` (e.g. the agent's own footprint removed from the map)
    pub penalty_override: Option<&'a HashMap<Cell, u32>>,
    /// safety cap on expansions
    pub max_expansions: usize,
}

#[derive(Debug, Clone)]
pub struct PlanResult {
    pub path: Vec<Cell>,
    pub cost: u32,
    pub expansions: usize,
    pub reached_goal: bool,
}

/// Space-time A*. Returns a path indexed by timestep (path[0] == start).
pub fn plan(q: &Query) -> Option<PlanResult> {
    if q.h[q.start] == u32::MAX {
        return None;
    }
    if q.res.vertex_blocked(q.start, 0) && q.res.vertex.contains(&(q.start, 0)) {
        // start occupied by a higher-priority agent at t=0 is a caller bug; treat as infeasible
        return None;
    }
    let t_floor = q.res.max_t();
    let deadline = q
        .horizon
        .unwrap_or(t_floor + (q.grid.tiles.len() as u32) * 2 + 8);

    // node arena
    let mut cells: Vec<Cell> = Vec::with_capacity(4096);
    let mut times: Vec<u32> = Vec::with_capacity(4096);
    let mut parent: Vec<u32> = Vec::with_capacity(4096);
    let mut g_cost: Vec<u32> = Vec::with_capacity(4096);
    let mut best_g: HashMap<(Cell, u32), u32> = HashMap::new();
    let mut open: BinaryHeap<Reverse<(u32, u32, u32)>> = BinaryHeap::new(); // (f, h, idx)

    let hscale = |c: Cell| q.h[c].saturating_mul(UNIT);
    cells.push(q.start);
    times.push(0);
    parent.push(u32::MAX);
    g_cost.push(0);
    best_g.insert((q.start, 0), 0);
    open.push(Reverse((hscale(q.start), hscale(q.start), 0)));

    let mut expansions = 0usize;
    while let Some(Reverse((_f, _h, idx))) = open.pop() {
        let idx = idx as usize;
        let (c, t, g) = (cells[idx], times[idx], g_cost[idx]);
        if best_g.get(&(c, t)).is_some_and(|&b| b < g) {
            continue;
        }
        let at_goal = c == q.goal && q.res.can_rest(c, t);
        let at_horizon = q.horizon.is_some_and(|hz| t >= hz);
        if at_goal || at_horizon {
            let mut path = Vec::with_capacity(t as usize + 1);
            let mut i = idx;
            while i != u32::MAX as usize {
                path.push(cells[i]);
                i = parent[i] as usize;
            }
            path.reverse();
            return Some(PlanResult {
                path,
                cost: g,
                expansions,
                reached_goal: c == q.goal,
            });
        }
        expansions += 1;
        if expansions > q.max_expansions || t >= deadline {
            if expansions > q.max_expansions {
                return None;
            }
            continue;
        }
        let nt = t + 1;
        let succ = std::iter::once(c).chain(q.grid.neighbors(c));
        for n in succ {
            if q.h[n] == u32::MAX
                || q.res.vertex_blocked(n, nt)
                || (n != c && q.res.edge_blocked(c, n, nt))
            {
                continue;
            }
            let pen = match (q.penalty, q.penalty_override.and_then(|o| o.get(&n))) {
                (Some(_), Some(&v)) => v,
                (Some(p), None) => p[n],
                _ => 0,
            };
            let step = UNIT + pen;
            let ng = g + step;
            let key = (n, nt);
            if best_g.get(&key).is_some_and(|&b| b <= ng) {
                continue;
            }
            best_g.insert(key, ng);
            let h = hscale(n);
            cells.push(n);
            times.push(nt);
            parent.push(idx as u32);
            g_cost.push(ng);
            open.push(Reverse((ng + h, h, (cells.len() - 1) as u32)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q<'a>(g: &'a Grid, s: Cell, goal: Cell, h: &'a [u32], r: &'a Reservations) -> Query<'a> {
        Query {
            grid: g,
            start: s,
            goal,
            h,
            res: r,
            horizon: None,
            penalty: None,
            penalty_override: None,
            max_expansions: 1_000_000,
        }
    }

    #[test]
    fn straight_line() {
        let g = Grid::from_ascii(".....");
        let h = g.bfs(4);
        let r = Reservations::new();
        let p = plan(&q(&g, 0, 4, &h, &r)).unwrap();
        assert_eq!(p.path, vec![0, 1, 2, 3, 4]);
        assert_eq!(p.cost, 4 * UNIT);
    }

    #[test]
    fn waits_for_reserved_cell() {
        let g = Grid::from_ascii(".....");
        let h = g.bfs(4);
        let mut r = Reservations::new();
        r.reserve_vertex(2, 2); // someone sits in cell 2 at t=2
        let p = plan(&q(&g, 0, 4, &h, &r)).unwrap();
        assert_eq!(p.path.len(), 6, "must wait one step: {:?}", p.path);
        for (t, &c) in p.path.iter().enumerate() {
            assert!(!(c == 2 && t == 2));
        }
    }

    #[test]
    fn no_head_on_swap() {
        // other robot moves 0 -> 1 arriving t=1; we sit at 1 and want 0 on a 2-cell corridor.
        let g = Grid::from_ascii("..");
        let h = g.bfs(0);
        let mut r = Reservations::new();
        r.reserve_vertex(0, 0);
        r.reserve_edge(0, 1, 1);
        r.reserve_vertex(1, 1);
        // moving 1->0 at t=1 swaps with the other robot; staying at 1 collides with it: infeasible
        assert!(plan(&q(&g, 1, 0, &h, &r)).is_none());
    }

    #[test]
    fn windowed_returns_partial() {
        let g = Grid::from_ascii("..........");
        let h = g.bfs(9);
        let r = Reservations::new();
        let mut qq = q(&g, 0, 9, &h, &r);
        qq.horizon = Some(4);
        let p = plan(&qq).unwrap();
        assert_eq!(p.path, vec![0, 1, 2, 3, 4]);
        assert!(!p.reached_goal);
    }

    #[test]
    fn congestion_penalty_detours() {
        // two equal-length routes around a pillar; penalise the top one
        let g = Grid::from_ascii(
            "...
             .@.
             ...",
        );
        let (s, goal) = (g.cell(0, 1), g.cell(2, 1));
        let h = g.bfs(goal);
        let r = Reservations::new();
        let mut pen = vec![0u32; 9];
        pen[g.cell(1, 0)] = 500;
        let mut qq = q(&g, s, goal, &h, &r);
        qq.penalty = Some(&pen);
        let p = plan(&qq).unwrap();
        assert!(
            p.path.contains(&g.cell(1, 2)),
            "should route via bottom: {:?}",
            p.path
        );
    }
}
