//! # fleetflow-core
//!
//! Multi-robot warehouse fleet coordination:
//! * [`grid`]   – warehouse floor model and BFS distance tables
//! * [`astar`]  – space-time A* with vertex/edge reservation tables
//! * [`cbs`]    – Conflict-Based Search (optimal MAPF) and prioritized planning
//! * [`assign`] – Hungarian (LP-optimal) and greedy task assignment
//! * [`sim`]    – lifelong simulator with rolling-horizon replanning and a collision validator
//! * [`features`], [`cnn`] – learned congestion model inputs and native inference

pub mod assign;
pub mod astar;
pub mod cbs;
pub mod cnn;
pub mod features;
pub mod grid;
pub mod rng;
pub mod sim;

#[cfg(target_arch = "wasm32")]
mod wasm;
