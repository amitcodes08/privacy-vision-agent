import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import react from '@vitejs/plugin-react';
import { crx } from '@crxjs/vite-plugin';
import manifest from './manifest.json' with { type: 'json' };

export default defineConfig({
  plugins: [react(), crx({ manifest })],
  resolve: {
    alias: {
      '@shared': resolve(__dirname, '../shared'),
      '~': resolve(__dirname, 'src'),
    },
  },
  // transformers.js pulls in onnxruntime-web; keep it out of the dep
  // pre-bundler so the .wasm/.mjs assets are emitted verbatim.
  optimizeDeps: { exclude: ['@huggingface/transformers', 'onnxruntime-web'] },
  worker: { format: 'es' },
  server: {
    port: 5173,
    strictPort: true,
    hmr: {
      port: 5173,
    },
  },
  build: {
    target: 'esnext',
    sourcemap: true,
    rollupOptions: {
      input: {
        popup: resolve(__dirname, 'src/popup/index.html'),
        offscreen: resolve(__dirname, 'src/offscreen/index.html'),
        permission: resolve(__dirname, 'src/permission/index.html'),
      },
    },
  },
});
