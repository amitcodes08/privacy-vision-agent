/**
 * Vendor the onnxruntime-web WASM binary into the extension.
 *
 * Why this script exists: transformers.js sets
 *
 *   ONNX_ENV.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${version}/dist/`
 *
 * for every context that is not a ServiceWorkerGlobalScope (see
 * `@huggingface/transformers/src/backends/onnx.js`). onnxruntime then fetches
 * its wasm — and, for a string prefix, dynamic-imports the emscripten glue —
 * from that URL. Remotely hosted code, which Manifest V3 forbids outright and
 * which `script-src` in `extension_pages` cannot whitelist, so
 * `from_pretrained` throws before a single weight is read.
 *
 * Only the `.wasm` is copied. onnxruntime resolves for browsers to
 * `ort.bundle.min.mjs`, which *embeds* the emscripten glue; `model-loader.ts`
 * sets `wasmPaths` to the object form `{ wasm }` — no prefix, no `mjs` — which
 * is what keeps it on that embedded copy instead of importing glue over the
 * network. Vendoring `ort-wasm-simd-threaded.jsep.mjs` too would be 44 KB that
 * nothing ever loads. Copying from `@huggingface/transformers/dist` rather than
 * `onnxruntime-web/dist` keeps the byte content identical to what that CDN URL
 * would have returned.
 */
import { copyFile, mkdir, rm, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const CLIENT = resolve(HERE, '..');

const FILES = ['ort-wasm-simd-threaded.jsep.wasm'];
/** Vendored by an earlier revision, before the embedded glue was in use. */
const STALE = ['ort-wasm-simd-threaded.jsep.mjs'];

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

// `public/` is copied verbatim into the build, so a leftover here ships forever.
for (const dir of [target, join(CLIENT, 'dist', 'ort')]) {
  for (const file of STALE) await rm(join(dir, file), { force: true });
}
