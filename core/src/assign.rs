//! Task allocation.
//!
//! Matching idle robots to open tasks is a linear assignment problem: minimise
//! `sum_ij c_ij x_ij` subject to each robot taking at most one task and each task at most one robot.
//! Its LP relaxation is totally unimodular, so the LP optimum is integral; the Hungarian algorithm
//! (Kuhn–Munkres, O(n^2 m)) solves it exactly. A greedy nearest-task baseline is kept for comparison.

/// Optimal assignment for an `n x m` cost matrix with `n <= m`.
/// Returns `row -> column` for every row.
pub fn hungarian(cost: &[Vec<i64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return vec![];
    }
    let m = cost[0].len();
    assert!(n <= m, "hungarian: need rows <= cols (got {n}x{m})");
    const INF: i64 = i64::MAX / 4;
    // 1-indexed potentials (e-maxx formulation)
    let mut u = vec![0i64; n + 1];
    let mut v = vec![0i64; m + 1];
    let mut p = vec![0usize; m + 1]; // p[j] = row matched to column j
    let mut way = vec![0usize; m + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![INF; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0;
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut ans = vec![usize::MAX; n];
    for j in 1..=m {
        if p[j] != 0 {
            ans[p[j] - 1] = j - 1;
        }
    }
    ans
}

/// Each robot in turn grabs its cheapest remaining task.
pub fn greedy(cost: &[Vec<i64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return vec![];
    }
    let m = cost[0].len();
    let mut taken = vec![false; m];
    let mut ans = vec![usize::MAX; n];
    for i in 0..n {
        let best = (0..m).filter(|&j| !taken[j]).min_by_key(|&j| cost[i][j]);
        if let Some(j) = best {
            taken[j] = true;
            ans[i] = j;
        }
    }
    ans
}

pub fn total(cost: &[Vec<i64>], a: &[usize]) -> i64 {
    a.iter()
        .enumerate()
        .filter(|(_, &j)| j != usize::MAX)
        .map(|(i, &j)| cost[i][j])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(cost: &[Vec<i64>]) -> i64 {
        fn rec(i: usize, cost: &[Vec<i64>], used: &mut Vec<bool>) -> i64 {
            if i == cost.len() {
                return 0;
            }
            let mut best = i64::MAX;
            for j in 0..cost[0].len() {
                if !used[j] {
                    used[j] = true;
                    best = best.min(cost[i][j] + rec(i + 1, cost, used));
                    used[j] = false;
                }
            }
            best
        }
        rec(0, cost, &mut vec![false; cost[0].len()])
    }

    #[test]
    fn matches_brute_force() {
        let mut rng = crate::rng::Rng::new(42);
        for _ in 0..200 {
            let n = 1 + rng.below(5);
            let m = n + rng.below(3);
            let cost: Vec<Vec<i64>> = (0..n)
                .map(|_| (0..m).map(|_| rng.below(50) as i64).collect())
                .collect();
            let a = hungarian(&cost);
            let mut seen = std::collections::HashSet::new();
            assert!(
                a.iter().all(|&j| seen.insert(j)),
                "columns must be distinct"
            );
            assert_eq!(total(&cost, &a), brute(&cost));
            assert!(total(&cost, &greedy(&cost)) >= total(&cost, &a));
        }
    }

    #[test]
    fn greedy_can_be_suboptimal() {
        let cost = vec![vec![1, 2], vec![1, 100]];
        assert_eq!(total(&cost, &greedy(&cost)), 101);
        assert_eq!(total(&cost, &hungarian(&cost)), 3);
    }
}
