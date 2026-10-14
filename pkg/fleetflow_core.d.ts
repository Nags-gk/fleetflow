/* tslint:disable */
/* eslint-disable */

export class WasmSim {
    free(): void;
    [Symbol.dispose](): void;
    congestion(): Float32Array;
    features(): Float32Array;
    height(): number;
    /**
     * Load the congestion CNN (JSON weights) for in-WASM inference (fallback when ONNX Runtime
     * Web is unavailable).
     */
    load_model(json: string): void;
    metrics_json(): string;
    needs_replan(): boolean;
    constructor(config_json: string);
    /**
     * 0 idle, 1 to pickup, 2 carrying to station
     */
    phases(): Uint8Array;
    /**
     * next planned cells, flattened with stride `replan_every`
     */
    plans(): Uint32Array;
    positions(): Uint32Array;
    /**
     * Run the native CNN on the current state and install the prediction as the cost layer.
     */
    predict_native(): boolean;
    reset_metrics(): void;
    set_congestion(map: Float32Array): void;
    /**
     * "none" | "density" | "external"
     */
    set_congestion_mode(mode: string, weight: number): void;
    step(n: number): void;
    /**
     * -1 when idle
     */
    targets(): Int32Array;
    /**
     * 0 free, 1 shelf, 2 station
     */
    tiles(): Uint8Array;
    width(): number;
}

/**
 * One-shot MAPF on the demo map. `agents_json`: [[start, goal], ...]; `algo`: "cbs" | "prioritized".
 */
export function solve_mapf(config_json: string, agents_json: string, algo: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmsim_free: (a: number, b: number) => void;
    readonly solve_mapf: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly wasmsim_congestion: (a: number) => [number, number];
    readonly wasmsim_features: (a: number) => [number, number];
    readonly wasmsim_height: (a: number) => number;
    readonly wasmsim_load_model: (a: number, b: number, c: number) => [number, number];
    readonly wasmsim_metrics_json: (a: number) => [number, number];
    readonly wasmsim_needs_replan: (a: number) => number;
    readonly wasmsim_new: (a: number, b: number) => [number, number, number];
    readonly wasmsim_phases: (a: number) => [number, number];
    readonly wasmsim_plans: (a: number) => [number, number];
    readonly wasmsim_positions: (a: number) => [number, number];
    readonly wasmsim_predict_native: (a: number) => number;
    readonly wasmsim_reset_metrics: (a: number) => void;
    readonly wasmsim_set_congestion: (a: number, b: number, c: number) => void;
    readonly wasmsim_set_congestion_mode: (a: number, b: number, c: number, d: number) => void;
    readonly wasmsim_step: (a: number, b: number) => void;
    readonly wasmsim_targets: (a: number) => [number, number];
    readonly wasmsim_tiles: (a: number) => [number, number];
    readonly wasmsim_width: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
