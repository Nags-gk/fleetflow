//! WebAssembly bindings for the browser demo.

use crate::cbs::{self, Agent};
use crate::features;
use crate::grid::Tile;
use crate::sim::{Config, CongestionSource, Phase, Sim};
use wasm_bindgen::prelude::*;

fn now() -> f64 {
    js_sys::Date::now()
}

#[wasm_bindgen]
pub struct WasmSim {
    sim: Sim,
    model: Option<crate::cnn::Cnn>,
}

#[wasm_bindgen]
impl WasmSim {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmSim, JsValue> {
        let cfg: Config =
            serde_json::from_str(config_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut sim = Sim::new(cfg);
        sim.now_ms = now;
        Ok(WasmSim { sim, model: None })
    }

    pub fn step(&mut self, n: u32) {
        self.sim.run(n);
    }

    pub fn needs_replan(&self) -> bool {
        self.sim.needs_replan()
    }

    pub fn width(&self) -> usize {
        self.sim.grid.width
    }

    pub fn height(&self) -> usize {
        self.sim.grid.height
    }

    /// 0 free, 1 shelf, 2 station
    pub fn tiles(&self) -> Vec<u8> {
        self.sim
            .grid
            .tiles
            .iter()
            .map(|t| match t {
                Tile::Free => 0,
                Tile::Shelf => 1,
                Tile::Station => 2,
            })
            .collect()
    }

    pub fn positions(&self) -> Vec<u32> {
        self.sim.robots.iter().map(|r| r.pos as u32).collect()
    }

    /// -1 when idle
    pub fn targets(&self) -> Vec<i32> {
        self.sim
            .robots
            .iter()
            .map(|r| r.target().map_or(-1, |t| t as i32))
            .collect()
    }

    /// 0 idle, 1 to pickup, 2 carrying to station
    pub fn phases(&self) -> Vec<u8> {
        self.sim
            .robots
            .iter()
            .map(|r| match r.task {
                None => 0,
                Some((_, Phase::ToPickup, _)) => 1,
                Some(_) => 2,
            })
            .collect()
    }

    /// next planned cells, flattened with stride `replan_every`
    pub fn plans(&self) -> Vec<u32> {
        self.sim
            .robots
            .iter()
            .flat_map(|r| r.plan.iter().map(|&c| c as u32).collect::<Vec<_>>())
            .collect()
    }

    pub fn metrics_json(&self) -> String {
        serde_json::to_string(&self.sim.metrics).unwrap()
    }

    pub fn features(&mut self) -> Vec<f32> {
        features::encode(&mut self.sim)
    }

    pub fn set_congestion(&mut self, map: Vec<f32>) {
        if map.len() == self.sim.congestion.len() {
            self.sim.congestion = map;
        }
    }

    pub fn congestion(&self) -> Vec<f32> {
        self.sim.congestion.clone()
    }

    /// "none" | "density" | "external"
    pub fn set_congestion_mode(&mut self, mode: &str, weight: f32) {
        self.sim.cfg.congestion = match mode {
            "density" => CongestionSource::Density,
            "external" => CongestionSource::External,
            _ => CongestionSource::None,
        };
        self.sim.cfg.congestion_weight = weight;
    }

    /// Load the congestion CNN (JSON weights) for in-WASM inference (fallback when ONNX Runtime
    /// Web is unavailable).
    pub fn load_model(&mut self, json: &str) -> Result<(), JsValue> {
        self.model =
            Some(crate::cnn::Cnn::from_json(json).map_err(|e| JsValue::from_str(&e.to_string()))?);
        Ok(())
    }

    /// Run the native CNN on the current state and install the prediction as the cost layer.
    pub fn predict_native(&mut self) -> bool {
        let Some(m) = self.model.as_ref() else {
            return false;
        };
        let x = features::encode(&mut self.sim);
        let (h, w) = (self.sim.grid.height, self.sim.grid.width);
        self.sim.congestion = m.forward(&x, h, w);
        true
    }

    pub fn reset_metrics(&mut self) {
        self.sim.reset_metrics();
    }
}

/// One-shot MAPF on the demo map. `agents_json`: [[start, goal], ...]; `algo`: "cbs" | "prioritized".
#[wasm_bindgen]
pub fn solve_mapf(config_json: &str, agents_json: &str, algo: &str) -> Result<String, JsValue> {
    let cfg: Config =
        serde_json::from_str(config_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let grid = crate::grid::Grid::warehouse(cfg.width, cfg.height, cfg.block_len);
    let pairs: Vec<(usize, usize)> =
        serde_json::from_str(agents_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let agents: Vec<Agent> = pairs
        .into_iter()
        .map(|(start, goal)| Agent { start, goal })
        .collect();
    let t0 = now();
    let sol = match algo {
        "cbs" => cbs::cbs(&grid, &agents, 5_000),
        _ => cbs::prioritized(&grid, &agents),
    };
    let ms = now() - t0;
    Ok(match sol {
        Some(s) => serde_json::json!({ "ok": true, "ms": ms, "solution": s }).to_string(),
        None => serde_json::json!({ "ok": false, "ms": ms }).to_string(),
    })
}
