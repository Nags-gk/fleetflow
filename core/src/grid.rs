//! Warehouse floor model: a 4-connected occupancy grid with shelves, aisles and pick stations.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub type Cell = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tile {
    Free,
    Shelf,
    Station,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub tiles: Vec<Tile>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Grid {
            width,
            height,
            tiles: vec![Tile::Free; width * height],
        }
    }

    /// Parse an ASCII map: `.` free, `@`/`#` shelf, `S` station.
    pub fn from_ascii(s: &str) -> Self {
        let rows: Vec<&str> = s.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        let height = rows.len();
        let width = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut g = Grid::new(width, height);
        for (y, row) in rows.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                g.tiles[y * width + x] = match ch {
                    '@' | '#' => Tile::Shelf,
                    'S' => Tile::Station,
                    _ => Tile::Free,
                };
            }
        }
        g
    }

    /// Kiva-style layout: shelf pods in 2xL blocks separated by `aisle`-wide aisles,
    /// pick stations on both side walls.
    pub fn warehouse(width: usize, height: usize, block_len: usize) -> Self {
        Self::warehouse_with_aisles(width, height, block_len, 2)
    }

    pub fn warehouse_with_aisles(
        width: usize,
        height: usize,
        block_len: usize,
        aisle: usize,
    ) -> Self {
        let mut g = Grid::new(width, height);
        let x0 = 4; // buffer zone in front of the stations
        let mut y = aisle;
        while y + 2 + aisle <= height {
            let mut x = x0;
            while x + block_len + 4 <= width {
                for dy in 0..2 {
                    for dx in 0..block_len {
                        g.tiles[(y + dy) * width + x + dx] = Tile::Shelf;
                    }
                }
                x += block_len + aisle;
            }
            y += 2 + aisle;
        }
        // pick stations on both side walls, each facing a 3-4 cell buffer zone
        for y in (1..height - 1).step_by(3) {
            g.tiles[y * width] = Tile::Station;
            g.tiles[y * width + width - 1] = Tile::Station;
        }
        g
    }

    #[inline]
    pub fn xy(&self, c: Cell) -> (usize, usize) {
        (c % self.width, c / self.width)
    }

    #[inline]
    pub fn cell(&self, x: usize, y: usize) -> Cell {
        y * self.width + x
    }

    #[inline]
    pub fn passable(&self, c: Cell) -> bool {
        self.tiles[c] != Tile::Shelf
    }

    /// 4-connected passable neighbours.
    pub fn neighbors(&self, c: Cell) -> impl Iterator<Item = Cell> + '_ {
        let (x, y) = self.xy(c);
        let w = self.width;
        let h = self.height;
        let cands = [
            if x > 0 { Some(c - 1) } else { None },
            if x + 1 < w { Some(c + 1) } else { None },
            if y > 0 { Some(c - w) } else { None },
            if y + 1 < h { Some(c + w) } else { None },
        ];
        cands
            .into_iter()
            .flatten()
            .filter(move |&n| self.passable(n))
    }

    pub fn stations(&self) -> Vec<Cell> {
        (0..self.tiles.len())
            .filter(|&c| self.tiles[c] == Tile::Station)
            .collect()
    }

    /// Free cells directly adjacent to a shelf: where a robot docks under / beside a pod.
    pub fn shelf_access_points(&self) -> Vec<Cell> {
        (0..self.tiles.len())
            .filter(|&c| self.tiles[c] == Tile::Free)
            .filter(|&c| {
                let (x, y) = self.xy(c);
                let w = self.width;
                [
                    (x > 0).then(|| c - 1),
                    (x + 1 < w).then(|| c + 1),
                    (y > 0).then(|| c - w),
                    (y + 1 < self.height).then(|| c + w),
                ]
                .into_iter()
                .flatten()
                .any(|n| self.tiles[n] == Tile::Shelf)
            })
            .collect()
    }

    pub fn free_cells(&self) -> Vec<Cell> {
        (0..self.tiles.len())
            .filter(|&c| self.tiles[c] == Tile::Free)
            .collect()
    }

    /// Exact shortest-path distances from `src` (BFS). `u32::MAX` = unreachable.
    pub fn bfs(&self, src: Cell) -> Vec<u32> {
        let mut d = vec![u32::MAX; self.tiles.len()];
        let mut q = VecDeque::new();
        d[src] = 0;
        q.push_back(src);
        while let Some(c) = q.pop_front() {
            for n in self.neighbors(c) {
                if d[n] == u32::MAX {
                    d[n] = d[c] + 1;
                    q.push_back(n);
                }
            }
        }
        d
    }
}

/// Lazily cached BFS distance tables, keyed by goal cell (true-distance heuristic for A*).
#[derive(Default, Clone)]
pub struct DistCache {
    tables: std::collections::HashMap<Cell, std::rc::Rc<Vec<u32>>>,
}

impl DistCache {
    pub fn get(&mut self, grid: &Grid, goal: Cell) -> std::rc::Rc<Vec<u32>> {
        self.tables
            .entry(goal)
            .or_insert_with(|| std::rc::Rc::new(grid.bfs(goal)))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_roundtrip_and_bfs() {
        let g = Grid::from_ascii(
            "....
             .@@.
             S...",
        );
        assert_eq!((g.width, g.height), (4, 3));
        assert!(!g.passable(g.cell(1, 1)));
        let d = g.bfs(g.cell(0, 0));
        assert_eq!(d[g.cell(3, 2)], 5);
        assert_eq!(g.stations(), vec![g.cell(0, 2)]);
    }

    #[test]
    fn warehouse_is_connected() {
        let g = Grid::warehouse(32, 20, 5);
        let s = g.stations()[0];
        let d = g.bfs(s);
        for c in g.free_cells() {
            assert_ne!(d[c], u32::MAX, "cell {c} unreachable");
        }
        assert!(!g.shelf_access_points().is_empty());
    }
}
