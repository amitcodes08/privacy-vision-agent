/**
 * Vendor the onnxruntime-web WASM assets into the extension.
 *
 * Why this script exists: transformers.js sets
 *
 *   ONNX_ENV.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${version}/dist/`
 *
 * for every context that is not a ServiceWorkerGlobalScope (see
 * `@huggingface/transformers/src/backends/onnx.js`). onnxruntime then
 * dynamic-imports `ort-wasm-simd-threaded.jsep.mjs` from that URL — remotely
 * hosted code, which Manifest V3 forbids outright and which `script-src` in
 * `extension_pages` cannot whitelist. The import is blocked, so
 * `from_pretrained` throws before a single weight is read. WebGPU does not save
 * it either: onnxruntime's WebGPU execution provider lives *inside* that same
 * jsep module.
 *
 * So we copy the two files the CDN would have served into `public/ort/`, and
 * `model-loader.ts` points `wasmPaths` at the extension's own origin. Copying
 * from `@huggingface/transformers/dist` rather than `onnxruntime-web/dist`
 * keeps the build byte-identical to what that URL would have returned.
 */
import { copyFile, mkdir, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const CLIENT = resolve(HERE, '..');

const FILES = ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm'];

/** node_modules may be hoisted to the workspace root. */
async function findSource() {
  for (const base of [CLIENT, resolve(CLIENT, '..')]) {
    const dir = join(base, 'node_modules', '@huggingface', 'transformers', 'dist');
    try {
      await stat(join(dir, FILES[0]));
      return dir;
    } catch {
      // keep looking
    }
  }
  throw new Error(
    'Could not find @huggingface/transformers/dist. Run `npm install` from the workspace root first.',
  );
}

const source = await findSource();
const target = join(CLIENT, 'public', 'ort');
await mkdir(target, { recursive: true });

for (const file of FILES) {
  await copyFile(join(source, file), join(target, file));
  const { size } = await stat(join(target, file));
  console.log(`[copy-ort] public/ort/${file} (${(size / 1024 / 1024).toFixed(1)} MB)`);
}
