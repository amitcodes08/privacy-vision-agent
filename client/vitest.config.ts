import { defineConfig } from 'vitest/config';
import { resolve } from 'node:path';

// Standalone config: the CRXJS plugin is deliberately absent so unit tests
// exercise the pure modules without an extension build.
export default defineConfig({
  resolve: {
    alias: {
      '@shared': resolve(import.meta.dirname, '../shared'),
      '~': resolve(import.meta.dirname, 'src'),
    },
  },
  test: {
    environment: 'jsdom',
    include: ['tests/**/*.test.ts'],
    globals: true,
  },
});
