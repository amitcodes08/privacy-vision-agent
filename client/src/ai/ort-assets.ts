/**
 * Where onnxruntime is allowed to get its WASM, and in which shape.
 *
 * Split out of `model-loader.ts` so the policy can be asserted without importing
 * transformers.js — and because the shape below is not a detail. Getting it
 * wrong cost two separate debugging sessions.
 *
 * Two things go wrong if this is left alone.
 *
 * 1. transformers.js sets `wasmPaths` to
 *    `https://cdn.jsdelivr.net/npm/@huggingface/transformers@<v>/dist/` for any
 *    context that is not a ServiceWorkerGlobalScope. That is remotely hosted
 *    code: Manifest V3 forbids it and `script-src` in `extension_pages` cannot
 *    whitelist a remote host, so `from_pretrained` threw before reading a
 *    single weight.
 *
 * 2. The obvious repair — repointing `wasmPaths` at `chrome-extension://…/ort/`
 *    — swaps that failure for a subtler one. `wasmPaths` is a union: a **string
 *    prefix** or an object `{mjs, wasm}`. onnxruntime resolves for browsers to
 *    `dist/ort.bundle.min.mjs`, the flavour that *embeds* the emscripten module
 *    factory precisely so nothing has to be fetched — but it only uses that
 *    embedded factory when neither a prefix nor an `mjs` override is set:
 *
 *      if (!mjsUrl && !prefix && embedded && scriptUrl && sameOrigin(scriptUrl))
 *        return [undefined, embedded];
 *      // …otherwise: (await import(prefix + 'ort-wasm-simd-threaded.jsep.mjs')).default
 *
 *    A string prefix therefore forces a dynamic import, and when that import
 *    yields no `default` the factory is `undefined` and calling it throws
 *    `TypeError: f is not a function` — which onnxruntime reports as
 *    `no available backend found. ERR: [webgpu] …`, sending everyone looking at
 *    the GPU for a failure that happened before WebGPU was ever reached.
 *
 * So: the object form, `wasm` only. The embedded factory is used, no `.mjs` is
 * ever imported, and `locateFile` fetches this one binary. It is the same shape
 * onnxruntime itself uses when it hands `wasmPaths` to its proxy worker.
 */

/**
 * `/ort/…` is deliberately origin-absolute: it resolves against whatever origin
 * is serving this module, which is correct both for a built extension
 * (`chrome-extension://…/ort/`) and for the crxjs dev server
 * (`http://localhost:5173/ort/`, served straight out of `public/`).
 * `npm run copy:ort` puts the file there; it runs as part of dev and build.
 *
 * chrome.* APIs are not exposed inside a dedicated worker, hence `import.meta`
 * rather than `chrome.runtime.getURL`.
 */
export const ORT_WASM_URL = new URL('/ort/ort-wasm-simd-threaded.jsep.wasm', import.meta.url).href;

export interface OrtWasmConfig {
  /** Object form, `wasm` only — see the note above. Never a string. */
  wasmPaths: { wasm: string };
  numThreads: number;
}

export function ortWasmConfig(): OrtWasmConfig {
  return {
    wasmPaths: { wasm: ORT_WASM_URL },
    // One thread, unconditionally. The embedded glue spawns its pthread pool
    // from its own `import.meta.url` — which, once onnxruntime is bundled into
    // our worker chunk, is that chunk rather than a standalone emscripten
    // module, so every pooled worker would re-run the bundle. WebGPU does the
    // heavy lifting anyway; on the wasm fallback this costs throughput and buys
    // a load that actually completes.
    numThreads: 1,
  };
}
