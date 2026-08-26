import { describe, expect, it } from 'vitest';
import { dedupeBoxes, redactFrame } from '~/privacy/canvas-redactor';

describe('dedupeBoxes', () => {
  it('drops boxes fully contained in another', () => {
    const outer = { x: 0, y: 0, width: 100, height: 100 };
    const inner = { x: 10, y: 10, width: 20, height: 20 };
    expect(dedupeBoxes([outer, inner])).toEqual([outer]);
    expect(dedupeBoxes([inner, outer])).toEqual([outer]);
  });

  it('keeps disjoint boxes and discards degenerate ones', () => {
    const a = { x: 0, y: 0, width: 10, height: 10 };
    const b = { x: 50, y: 50, width: 10, height: 10 };
    expect(dedupeBoxes([a, b, { x: 5, y: 5, width: 0, height: 4 }])).toEqual([a, b]);
  });
});

// OffscreenCanvas is unavailable in jsdom and in plain Node; these run in a
// browser-backed vitest environment or Chrome. Kept in-repo so the perf
// budget is asserted where it can be.
const hasCanvas = typeof OffscreenCanvas !== 'undefined';

describe.skipIf(!hasCanvas)('redactFrame', () => {
  const makeSource = (w = 1280, h = 800) => {
    const c = new OffscreenCanvas(w, h);
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#f00';
    ctx.fillRect(100, 100, 200, 40);
    return c;
  };

  it('paints every box and stays under the 30ms budget', async () => {
    const result = await redactFrame(makeSource(), {
      boxes: [{ x: 100, y: 100, width: 200, height: 40 }],
      scale: 1,
      style: 'black',
    });
    expect(result.boxesApplied).toBe(1);
    expect(result.base64.length).toBeGreaterThan(100);
    expect(result.elapsedMs).toBeLessThan(30);
  });

  it('skips boxes that fall outside the frame', async () => {
    const result = await redactFrame(makeSource(320, 200), {
      boxes: [{ x: 5000, y: 5000, width: 10, height: 10 }],
      scale: 1,
    });
    expect(result.boxesApplied).toBe(0);
  });
});
