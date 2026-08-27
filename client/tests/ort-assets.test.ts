import { describe, expect, it } from 'vitest';
import { ORT_WASM_URL, ortWasmConfig } from '~/ai/ort-assets';

/**
 * These look like tautologies and are not. `wasmPaths` accepts either a string
 * prefix or `{mjs, wasm}`, and the two take different code paths inside
 * onnxruntime: the string form makes it dynamic-import the emscripten glue,
 * abandoning the copy already embedded in `ort.bundle.min.mjs`. When that import
 * resolves to something without a `default`, the factory is `undefined` and
 * onnxruntime reports `no available backend found. ERR: [webgpu] TypeError: f is
 * not a function` — a WebGPU-shaped message for a failure that never reached
 * WebGPU. The string form is the natural thing to write, so it is worth a test.
 */
describe('ortWasmConfig', () => {
  it('uses the object form, so the embedded emscripten glue stays in play', () => {
    const { wasmPaths } = ortWasmConfig();
    expect(typeof wasmPaths).toBe('object');
    expect(wasmPaths.wasm).toBe(ORT_WASM_URL);
  });

  it('sets no mjs override — that would force the dynamic import too', () => {
    expect(Object.keys(ortWasmConfig().wasmPaths)).toEqual(['wasm']);
  });

  it('points at the vendored binary, not at a CDN', () => {
    expect(ORT_WASM_URL).toMatch(/\/ort\/ort-wasm-simd-threaded\.jsep\.wasm$/);
    expect(ORT_WASM_URL).not.toMatch(/cdn|jsdelivr|unpkg|huggingface\.co/i);
  });

  it('asks for a single thread, so no pthread pool is spawned', () => {
    expect(ortWasmConfig().numThreads).toBe(1);
  });
});
