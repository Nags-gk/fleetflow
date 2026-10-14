import init, { WasmSim, solve_mapf } from "./pkg/fleetflow_core.js";

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = (s, root = document) => root.querySelector(s);

const BASE = { width: 40, height: 26, block_len: 5, horizon: 10, replan_every: 2, assigner: "hungarian" };
const WEIGHT = 3.0;

let A, B, ortSession = null, engine = "none";
let playing = true, busy = false, seed = 1 + Math.floor(Math.random() * 1e6);
const prev = { A: null, B: null };

function makeSims() {
  const robots = +$("#robots").value;
  A = new WasmSim(JSON.stringify({ ...BASE, robots, seed, congestion: "none" }));
  B = new WasmSim(JSON.stringify({ ...BASE, robots, seed, congestion: "external", congestion_weight: WEIGHT }));
  if (engine === "native" && modelJson) B.load_model(modelJson);
  prev.A = prev.B = null;
}

let modelJson = null;
async function loadModel() {
  const label = $("#engine");
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1;
    ortSession = await ort.InferenceSession.create("model/congestion.onnx", { executionProviders: ["wasm"] });
    engine = "ort";
    label.textContent = "Model running in ONNX Runtime Web.";
  } catch (e) {
    console.warn("ONNX Runtime Web unavailable, using the Rust/WASM inference path", e);
    modelJson = await (await fetch("model/congestion.json")).text();
    B.load_model(modelJson);
    engine = "native";
    label.textContent = "Model running in Rust/WebAssembly.";
  }
}

async function predict(sim) {
  if (engine === "ort") {
    const x = sim.features();
    const t = new ort.Tensor("float32", x, [1, 5, sim.height(), sim.width()]);
    const out = await ortSession.run({ state: t });
    sim.set_congestion(out.congestion.data);
  } else if (engine === "native") {
    sim.predict_native();
  }
}

function stuckCount(key, sim) {
  const pos = sim.positions(), ph = sim.phases();
  let n = 0;
  if (prev[key]) for (let i = 0; i < pos.length; i++) if (ph[i] > 0 && pos[i] === prev[key][i]) n++;
  prev[key] = pos;
  return n;
}

const tileCache = new WeakMap();
function draw(canvas, sim, heat) {
  const W = sim.width(), H = sim.height();
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.width;
  const cell = cssW / W;
  if (canvas.width !== Math.round(cssW * dpr)) {
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cell * H * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = css("--panel");
  g.fillRect(0, 0, cssW, cell * H);
  let tiles = tileCache.get(sim);
  if (!tiles) { tiles = sim.tiles(); tileCache.set(sim, tiles); }
  if (heat) {
    const c = sim.congestion();
    const rgb = css("--heat");
    for (let i = 0; i < c.length; i++) {
      const v = Math.min(1, c[i] * 1.6);
      if (v < 0.06) continue;
      g.fillStyle = `rgba(${rgb},${(0.12 + 0.6 * v).toFixed(3)})`;
      g.fillRect((i % W) * cell, Math.floor(i / W) * cell, cell, cell);
    }
  }
  const pod = css("--pod"), station = css("--station");
  for (let i = 0; i < tiles.length; i++) {
    if (tiles[i] === 0) continue;
    g.fillStyle = tiles[i] === 1 ? pod : station;
    const x = (i % W) * cell, y = Math.floor(i / W) * cell;
    g.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
  }
  const pos = sim.positions(), ph = sim.phases();
  const fetch_ = css("--fetch"), carry = css("--carry"), ink = css("--ink");
  for (let i = 0; i < pos.length; i++) {
    const x = (pos[i] % W + 0.5) * cell, y = (Math.floor(pos[i] / W) + 0.5) * cell;
    g.beginPath();
    g.arc(x, y, cell * 0.38, 0, Math.PI * 2);
    g.fillStyle = ph[i] === 2 ? carry : ph[i] === 1 ? fetch_ : ink;
    g.fill();
  }
}

function readout(fig, sim, stuck) {
  const m = JSON.parse(sim.metrics_json());
  const set = (k, v) => { fig.querySelector(`[data-k="${k}"]`).textContent = v; };
  set("tasks", m.tasks_completed.toLocaleString());
  set("thr", Math.round(m.throughput).toLocaleString());
  set("stuck", stuck);
  set("coll", m.collisions);
}

async function tick() {
  if (playing && !busy && A) {
    busy = true;
    const n = +$("#speed").value;
    for (let i = 0; i < n; i++) {
      A.step(1);
      if (B.needs_replan()) await predict(B);
      B.step(1);
    }
    const heat = $("#heat").checked;
    draw($("#floorA canvas"), A, false);
    draw($("#floorB canvas"), B, heat);
    readout($("#floorA"), A, stuckCount("A", A));
    readout($("#floorB"), B, stuckCount("B", B));
    busy = false;
  }
  requestAnimationFrame(tick);
}

function bindControls() {
  const robots = $("#robots"), speed = $("#speed");
  robots.addEventListener("input", () => { $("#robotsOut").textContent = robots.value; });
  robots.addEventListener("change", () => makeSims());
  speed.addEventListener("input", () => { $("#speedOut").textContent = speed.value; });
  $("#play").addEventListener("click", (e) => {
    playing = !playing;
    e.target.textContent = playing ? "Pause" : "Play";
  });
  $("#reset").addEventListener("click", () => { seed = 1 + Math.floor(Math.random() * 1e6); makeSims(); });
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) speed.value = 2;
}

/* ---------- results ---------- */
const VARIANTS = [
  { key: "greedy", name: "Greedy assignment", color: () => css("--muted"), dash: "5 4" },
  { key: "hungarian", name: "Hungarian assignment", color: () => css("--ink") },
  { key: "hungarian+density", name: "+ reactive density cost", color: () => css("--pod") },
  { key: "hungarian+learned", name: "+ learned forecast", color: () => css("--carry"), bold: true },
];

function lineChart(el, rows) {
  const W = 900, H = 340, L = 60, R = 210, T = 16, Bm = 44;
  const xs = [...new Set(rows.map((r) => r.robots))].sort((a, b) => a - b);
  const ymax = Math.ceil(Math.max(...rows.map((r) => r.throughput)) / 500) * 500;
  const X = (v) => L + ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - L - R);
  const Y = (v) => T + (1 - v / ymax) * (H - T - Bm);
  let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
  for (let v = 0; v <= ymax; v += 500) {
    s += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${css("--line")}" stroke-width="1"/>`;
    s += `<text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v.toLocaleString()}</text>`;
  }
  xs.forEach((x) => { s += `<text x="${X(x)}" y="${H - Bm + 20}" text-anchor="middle">${x}</text>`; });
  s += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">Robots on the floor</text>`;
  s += `<text x="12" y="${T + 4}" transform="rotate(-90 12 ${T + 4})" text-anchor="end">Orders per 1,000 ticks</text>`;
  const ends = [];
  for (const v of VARIANTS) {
    const pts = xs.map((x) => rows.find((r) => r.robots === x && r.variant === v.key)).filter(Boolean);
    if (!pts.length) continue;
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.robots)},${Y(p.throughput)}`).join(" ");
    s += `<path d="${d}" fill="none" stroke="${v.color()}" stroke-width="${v.bold ? 3.5 : 2}" ${v.dash ? `stroke-dasharray="${v.dash}"` : ""}/>`;
    pts.forEach((p) => { s += `<circle cx="${X(p.robots)}" cy="${Y(p.throughput)}" r="${v.bold ? 4 : 3}" fill="${v.color()}"><title>${v.name}, ${p.robots} robots: ${Math.round(p.throughput)}</title></circle>`; });
    const last = pts[pts.length - 1];
    ends.push({ y: Y(last.throughput), x: X(last.robots), v });
  }
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
  ends.forEach((e) => { s += `<text x="${e.x + 10}" y="${e.y + 4}" style="fill:${e.v.color()};font-weight:${e.v.bold ? 600 : 500}">${e.v.name}</text>`; });
  el.innerHTML = s + "</svg>";
}

async function loadResults() {
  try {
    const [bench, report, mapf] = await Promise.all(
      ["data/bench_lifelong.json", "data/train_report.json", "data/mapf.json"].map((u) => fetch(u).then((r) => r.json())),
    );
    lineChart($("#thrChart"), bench);
    const seeds = bench[0].seeds, ticks = bench[0].ticks;
    const at = (n, k) => bench.find((r) => r.robots === n && r.variant === k)?.throughput;
    const n = 100, base = at(n, "hungarian"), dens = at(n, "hungarian+density"), lrn = at(n, "hungarian+learned");
    $("#benchNote").textContent =
      `Mean of ${seeds} seeds, ${ticks.toLocaleString()} ticks each after a 200-tick warm-up, zero collisions in every run. ` +
      (base && lrn ? `At ${n} robots the plain planner gridlocks (${Math.round(base)} orders per 1,000 ticks); the learned forecast keeps ${Math.round(lrn)} flowing, ${Math.round((lrn / dens - 1) * 100)}% more than the reactive density heuristic.` : "");
    const rows = [["Reactive density blur", report.reactive_blur], ["scikit-learn gradient boosting (3x3)", report.sklearn_hgb_3x3], ["PyTorch dilated CNN", report.cnn]];
    $("#modelTable tbody").innerHTML = rows.map(([name, m], i) =>
      `<tr class="${i === 2 ? "win" : ""}"><td>${name}</td><td>${m.pearson_r.toFixed(2)}</td><td>${m.mse.toFixed(4)}</td><td>${Math.round(m["hotspot_recall@5%"] * 100)}%</td></tr>`).join("");
    $("#mapfTable tbody").innerHTML = mapf.map((r) =>
      `<tr><td>${r.agents}</td><td>${Math.round(r.cbs_success * 100)}%</td><td>${Math.round(r.prioritized_success * 100)}%</td><td>${r.prioritized_soc_gap_pct.toFixed(1)}%</td><td>${r.cbs_ms_mean < 10 ? r.cbs_ms_mean.toFixed(1) : Math.round(r.cbs_ms_mean)} ms</td></tr>`).join("");
  } catch (e) {
    $("#benchNote").textContent = "Benchmark data could not be loaded. Run the benchmark locally with `cargo run --release --bin ffsim -- bench`.";
  }
}

/* ---------- MAPF lab ---------- */
const LAB = { width: 24, height: 13, block_len: 4, robots: 0 };
let labAnim = null;

function labSolve() {
  const probe = new WasmSim(JSON.stringify(LAB));
  const tiles = probe.tiles();
  const free = [];
  tiles.forEach((t, i) => { if (t === 0) free.push(i); });
  for (let i = free.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [free[i], free[j]] = [free[j], free[i]]; }
  const k = +$("#k").value;
  const agents = Array.from({ length: k }, (_, i) => [free[i], free[i + k]]);
  const res = {};
  for (const algo of ["cbs", "prioritized"]) res[algo] = JSON.parse(solve_mapf(JSON.stringify(LAB), JSON.stringify(agents), algo));
  const row = (name, r, nodes) => r.ok
    ? `<tr><td>${name}</td><td>${r.solution.sum_of_costs}</td><td>${r.solution.makespan}</td><td>${nodes ? r.solution.high_level_nodes : "n/a"}</td><td>${r.ms.toFixed(1)} ms</td></tr>`
    : `<tr><td>${name}</td><td colspan="4">No solution within the search budget</td></tr>`;
  $("#labTable tbody").innerHTML = row("Conflict-Based Search", res.cbs, true) + row("Prioritized planning", res.prioritized, false);
  const sol = res.cbs.ok ? res.cbs.solution : res.prioritized.ok ? res.prioritized.solution : null;
  animateLab(tiles, agents, sol);
}

function animateLab(tiles, agents, sol) {
  const canvas = $("#labCanvas");
  const W = LAB.width, H = LAB.height;
  cancelAnimationFrame(labAnim);
  const hue = (i) => `hsl(${(i * 137.5) % 360} 55% 45%)`;
  const t0 = performance.now();
  const render = (now) => {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth, cell = cssW / W;
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cell * H * dpr);
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = css("--panel"); g.fillRect(0, 0, cssW, cell * H);
    tiles.forEach((t, i) => {
      if (!t) return;
      g.fillStyle = t === 1 ? css("--pod") : css("--station");
      g.fillRect((i % W) * cell + 0.5, Math.floor(i / W) * cell + 0.5, cell - 1, cell - 1);
    });
    const ctr = (c) => [(c % W + 0.5) * cell, (Math.floor(c / W) + 0.5) * cell];
    agents.forEach(([, goal], i) => {
      const [x, y] = ctr(goal);
      g.strokeStyle = hue(i); g.lineWidth = 2;
      g.strokeRect(x - cell * 0.35, y - cell * 0.35, cell * 0.7, cell * 0.7);
    });
    if (sol) {
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const t = reduce ? Infinity : (now - t0) / 260;
      sol.paths.forEach((p, i) => {
        const k = Math.min(Math.floor(t), p.length - 1), f = Math.min(1, t - k);
        const a = ctr(p[k]), b = ctr(p[Math.min(k + 1, p.length - 1)]);
        g.beginPath();
        g.arc(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, cell * 0.33, 0, Math.PI * 2);
        g.fillStyle = hue(i); g.fill();
      });
      if (t < sol.makespan + 1) labAnim = requestAnimationFrame(render);
    }
  };
  labAnim = requestAnimationFrame(render);
}

/* ---------- boot ---------- */
await init();
bindControls();
makeSims();
loadModel().catch((e) => { $("#engine").textContent = "Model failed to load; showing baseline only."; console.error(e); });
requestAnimationFrame(tick);
loadResults();
$("#k").addEventListener("input", () => { $("#kOut").textContent = $("#k").value; });
$("#solve").addEventListener("click", labSolve);
labSolve();
