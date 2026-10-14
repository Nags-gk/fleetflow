//! One-shot Multi-Agent Path Finding.
//!
//! * [`cbs`] — Conflict-Based Search (Sharon et al., 2015): optimal sum-of-costs. A best-first
//!   constraint tree (ties broken by fewest remaining conflicts); each node re-plans one agent
//!   with space-time A* under its constraints.
//! * [`prioritized`] — fast, suboptimal, incomplete baseline: agents plan in order and reserve
//!   their paths for everyone behind them.

use crate::astar::{plan, Query, Reservations};
use crate::grid::{Cell, DistCache, Grid};
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
pub struct Agent {
    pub start: Cell,
    pub goal: Cell,
}

#[derive(Debug, Clone, Serialize)]
pub struct Solution {
    pub paths: Vec<Vec<Cell>>,
    /// sum of path lengths in timesteps (until each agent rests at its goal)
    pub sum_of_costs: u32,
    pub makespan: u32,
    /// constraint-tree nodes expanded (CBS) or agents planned (prioritized)
    pub high_level_nodes: usize,
    pub low_level_expansions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conflict {
    Vertex {
        a: usize,
        b: usize,
        cell: Cell,
        t: u32,
    },
    /// a moves u->v while b moves v->u, both arriving at t
    Edge {
        a: usize,
        b: usize,
        u: Cell,
        v: Cell,
        t: u32,
    },
}

#[inline]
fn at(path: &[Cell], t: usize) -> Cell {
    path[t.min(path.len() - 1)]
}

/// First conflict in time order (agents rest at their goal after their path ends).
pub fn first_conflict(paths: &[Vec<Cell>]) -> Option<Conflict> {
    let horizon = paths.iter().map(Vec::len).max().unwrap_or(0);
    for t in 0..horizon {
        for a in 0..paths.len() {
            for b in (a + 1)..paths.len() {
                let (pa, pb) = (at(&paths[a], t), at(&paths[b], t));
                if pa == pb {
                    return Some(Conflict::Vertex {
                        a,
                        b,
                        cell: pa,
                        t: t as u32,
                    });
                }
                if t + 1 < horizon {
                    let (na, nb) = (at(&paths[a], t + 1), at(&paths[b], t + 1));
                    if pa == nb && pb == na && pa != na {
                        return Some(Conflict::Edge {
                            a,
                            b,
                            u: pa,
                            v: na,
                            t: (t + 1) as u32,
                        });
                    }
                }
            }
        }
    }
    None
}

/// Number of conflicting (agent pair, timestep) events: the CBS tie-breaker.
pub fn count_conflicts(paths: &[Vec<Cell>]) -> u32 {
    let horizon = paths.iter().map(Vec::len).max().unwrap_or(0);
    let mut n = 0;
    for t in 0..horizon {
        for a in 0..paths.len() {
            for b in (a + 1)..paths.len() {
                let (pa, pb) = (at(&paths[a], t), at(&paths[b], t));
                if pa == pb {
                    n += 1;
                } else if t + 1 < horizon {
                    let (na, nb) = (at(&paths[a], t + 1), at(&paths[b], t + 1));
                    if pa == nb && pb == na {
                        n += 1;
                    }
                }
            }
        }
    }
    n
}

fn cost_of(path: &[Cell]) -> u32 {
    // trailing waits at the goal are free
    let mut end = path.len();
    while end > 1 && path[end - 1] == path[end - 2] {
        end -= 1;
    }
    (end - 1) as u32
}

/// One child of a constraint-tree split: which agent gets which new constraint.
type Branch = (usize, Box<dyn Fn(&mut Reservations)>);

struct CtNode {
    constraints: Vec<Reservations>,
    paths: Vec<Vec<Cell>>,
    cost: u32,
}

fn low_level(
    grid: &Grid,
    agent: Agent,
    h: &[u32],
    cons: &Reservations,
    budget: usize,
) -> Option<(Vec<Cell>, usize)> {
    let q = Query {
        grid,
        start: agent.start,
        goal: agent.goal,
        h,
        res: cons,
        horizon: None,
        penalty: None,
        penalty_override: None,
        max_expansions: budget,
    };
    plan(&q).map(|r| (r.path, r.expansions))
}

/// Conflict-Based Search. Returns `None` if no solution is found within `max_nodes` CT expansions.
pub fn cbs(grid: &Grid, agents: &[Agent], max_nodes: usize) -> Option<Solution> {
    let mut dc = DistCache::default();
    let hs: Vec<_> = agents.iter().map(|a| dc.get(grid, a.goal)).collect();
    let budget = 200_000;
    let mut ll = 0usize;

    let mut root = CtNode {
        constraints: vec![Reservations::new(); agents.len()],
        paths: Vec::new(),
        cost: 0,
    };
    for (i, &a) in agents.iter().enumerate() {
        let (p, e) = low_level(grid, a, &hs[i], &root.constraints[i], budget)?;
        ll += e;
        root.paths.push(p);
    }
    root.cost = root.paths.iter().map(|p| cost_of(p)).sum();

    let mut arena = vec![root];
    let mut open = BinaryHeap::new();
    open.push(Reverse((
        arena[0].cost,
        count_conflicts(&arena[0].paths),
        0usize,
    )));
    let mut expanded = 0usize;

    while let Some(Reverse((_, _, idx))) = open.pop() {
        expanded += 1;
        if expanded > max_nodes {
            return None;
        }
        let conflict = first_conflict(&arena[idx].paths);
        let Some(conflict) = conflict else {
            let node = &arena[idx];
            return Some(Solution {
                paths: node.paths.clone(),
                sum_of_costs: node.cost,
                makespan: node.paths.iter().map(|p| cost_of(p)).max().unwrap_or(0),
                high_level_nodes: expanded,
                low_level_expansions: ll,
            });
        };
        let branches: [Branch; 2] = match conflict {
            Conflict::Vertex { a, b, cell, t } => [
                (
                    a,
                    Box::new(move |r: &mut Reservations| r.reserve_vertex(cell, t)),
                ),
                (
                    b,
                    Box::new(move |r: &mut Reservations| r.reserve_vertex(cell, t)),
                ),
            ],
            Conflict::Edge { a, b, u, v, t } => [
                (
                    a,
                    Box::new(move |r: &mut Reservations| r.forbid_move(u, v, t)),
                ),
                (
                    b,
                    Box::new(move |r: &mut Reservations| r.forbid_move(v, u, t)),
                ),
            ],
        };
        for (agent, add) in branches.iter() {
            let mut cons = arena[idx].constraints.clone();
            add(&mut cons[*agent]);
            let Some((p, e)) = low_level(grid, agents[*agent], &hs[*agent], &cons[*agent], budget)
            else {
                continue;
            };
            ll += e;
            let mut paths = arena[idx].paths.clone();
            paths[*agent] = p;
            let cost = paths.iter().map(|p| cost_of(p)).sum();
            let nc = count_conflicts(&paths);
            arena.push(CtNode {
                constraints: cons,
                paths,
                cost,
            });
            open.push(Reverse((cost, nc, arena.len() - 1)));
        }
    }
    None
}

/// Prioritized planning in the given agent order.
pub fn prioritized(grid: &Grid, agents: &[Agent]) -> Option<Solution> {
    let mut dc = DistCache::default();
    let mut res = Reservations::new();
    let mut paths = Vec::with_capacity(agents.len());
    let mut ll = 0;
    // every agent's start is occupied at t=0 until it leaves; reserve starts of later agents
    // for t=0 so earlier agents don't plan through them on the first step.
    for (i, a) in agents.iter().enumerate() {
        let h = dc.get(grid, a.goal);
        let mut local = res.clone();
        for b in &agents[i + 1..] {
            local.reserve_vertex(b.start, 0);
            local.reserve_vertex(b.start, 1);
        }
        let q = Query {
            grid,
            start: a.start,
            goal: a.goal,
            h: &h,
            res: &local,
            horizon: None,
            penalty: None,
            penalty_override: None,
            max_expansions: 200_000,
        };
        let r = plan(&q)?;
        ll += r.expansions;
        res.reserve_path(&r.path, true);
        paths.push(r.path);
    }
    if first_conflict(&paths).is_some() {
        return None;
    }
    Some(Solution {
        sum_of_costs: paths.iter().map(|p| cost_of(p)).sum(),
        makespan: paths.iter().map(|p| cost_of(p)).max().unwrap_or(0),
        paths,
        high_level_nodes: agents.len(),
        low_level_expansions: ll,
    })
}

/// Lower bound used in tests/benchmarks: sum of individual shortest-path distances.
pub fn sum_of_distances(grid: &Grid, agents: &[Agent]) -> u32 {
    agents.iter().map(|a| grid.bfs(a.goal)[a.start]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_in_corridor_with_bay_is_solved_optimally() {
        // classic: two agents swap ends of a corridor that has a single side bay
        let g = Grid::from_ascii(
            ".....
             @@.@@",
        );
        let agents = [Agent { start: 0, goal: 4 }, Agent { start: 4, goal: 0 }];
        let s = cbs(&g, &agents, 10_000).expect("solvable");
        assert!(first_conflict(&s.paths).is_none());
        // one agent ducks into the bay (+2 moves), the other waits once for it: 6 + 5 = 11
        assert_eq!(s.sum_of_costs, 11, "{:?}", s.paths);
    }

    #[test]
    fn cbs_never_worse_than_prioritized() {
        let g = Grid::warehouse(20, 11, 4);
        let free = g.free_cells();
        let mut rng = crate::rng::Rng::new(7);
        for _ in 0..10 {
            let mut cells = free.clone();
            rng.shuffle(&mut cells);
            let agents: Vec<Agent> = (0..6)
                .map(|i| Agent {
                    start: cells[i],
                    goal: cells[i + 6],
                })
                .collect();
            let c = cbs(&g, &agents, 20_000).expect("cbs");
            assert!(first_conflict(&c.paths).is_none());
            assert!(c.sum_of_costs >= sum_of_distances(&g, &agents));
            if let Some(p) = prioritized(&g, &agents) {
                assert!(c.sum_of_costs <= p.sum_of_costs);
            }
        }
    }

    #[test]
    fn detects_edge_conflict() {
        let paths = vec![vec![0, 1], vec![1, 0]];
        assert!(matches!(
            first_conflict(&paths),
            Some(Conflict::Edge { .. })
        ));
    }
}
