/**
 * Frame sanitizer. Runs in the service worker (OffscreenCanvas) or any
 * DOM context. Guarantees: the returned base64 has every supplied box
 * painted over destructively *before* encoding — the original pixels are
 * never present in the output buffer.
 */
import { DEFAULTS, type BoundingBox } from '@shared/types';

export type RedactionStyle = 'black' | 'blur' | 'pixelate';

export interface RedactOptions {
  boxes: BoundingBox[];
  /** Source scale factor: capture DPR vs CSS-pixel box coords. */
  scale?: number;
  maxWidth?: number;
  quality?: number;
  style?: RedactionStyle;
  /** Grow each box by N CSS px to cover antialiased text edges. */
  padding?: number;
  mime?: 'image/jpeg' | 'image/webp';
}

export interface RedactResult {
  /** base64 WITHOUT the `data:` prefix. */
  base64: string;
  mime: 'image/jpeg' | 'image/webp';
  width: number;
  height: number;
  bytes: number;
  boxesApplied: number;
  elapsedMs: number;
}

type Source = ImageBitmap | OffscreenCanvas | HTMLCanvasElement | ImageData;

export async function redactFrame(source: Source, opts: RedactOptions): Promise<RedactResult> {
  const t0 = now();
  const mime = opts.mime ?? 'image/jpeg';
  const quality = opts.quality ?? DEFAULTS.jpegQuality;
  const maxWidth = opts.maxWidth ?? DEFAULTS.maxFrameWidth;
  const padding = opts.padding ?? 2;
  const style = opts.style ?? 'black';

  const { width: srcW, height: srcH } = dimensions(source);
  if (srcW === 0 || srcH === 0) throw new Error('redactFrame: empty source');

  // Boxes are in CSS px; the capture may be at devicePixelRatio.
  const captureScale = opts.scale ?? srcW / Math.max(1, cssWidthHint(srcW));
  const outScale = Math.min(1, maxWidth / srcW);
  const outW = Math.max(1, Math.round(srcW * outScale));
  const outH = Math.max(1, Math.round(srcH * outScale));

  const canvas = new OffscreenCanvas(outW, outH);
  const ctx = canvas.getContext('2d', { alpha: false });
  if (!ctx) throw new Error('redactFrame: 2d context unavailable');

  ctx.drawImage(toDrawable(source), 0, 0, outW, outH);

  let applied = 0;
  const k = captureScale * outScale;
  for (const b of opts.boxes) {
    const x = Math.floor((b.x - padding) * k);
    const y = Math.floor((b.y - padding) * k);
    const w = Math.ceil((b.width + padding * 2) * k);
    const h = Math.ceil((b.height + padding * 2) * k);
    if (w <= 0 || h <= 0 || x > outW || y > outH || x + w < 0 || y + h < 0) continue;
    paint(ctx, style, clamp(x, 0, outW), clamp(y, 0, outH), Math.min(w, outW), Math.min(h, outH));
    applied++;
  }

  const blob = await canvas.convertToBlob({ type: mime, quality });
  const base64 = await blobToBase64(blob);
  return {
    base64,
    mime,
    width: outW,
    height: outH,
    bytes: blob.size,
    boxesApplied: applied,
    elapsedMs: now() - t0,
  };
}

function paint(
  ctx: OffscreenCanvasRenderingContext2D,
  style: RedactionStyle,
  x: number,
  y: number,
  w: number,
  h: number,
): void {
  if (style === 'black') {
    ctx.fillStyle = '#000';
    ctx.fillRect(x, y, w, h);
    return;
  }
  if (style === 'blur') {
    // Re-draw the region through a blur filter, then darken so residual
    // high-contrast glyph structure cannot be OCR'd back.
    ctx.save();
    ctx.filter = 'blur(12px)';
    ctx.drawImage(ctx.canvas, x, y, w, h, x, y, w, h);
    ctx.restore();
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(x, y, w, h);
    return;
  }
  // pixelate: downsample to ~6px blocks and scale back up.
  const bw = Math.max(1, Math.round(w / 6));
  const bh = Math.max(1, Math.round(h / 6));
  const tmp = new OffscreenCanvas(bw, bh);
  const tctx = tmp.getContext('2d');
  if (!tctx) return;
  tctx.drawImage(ctx.canvas, x, y, w, h, 0, 0, bw, bh);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, bw, bh, x, y, w, h);
  ctx.imageSmoothingEnabled = true;
}

/** Convert data URL to Blob without fetch (which is unsupported on data: in MV3 service workers). */
export function dataUrlToBlob(dataUrl: string): Blob {
  const comma = dataUrl.indexOf(',');
  const header = comma >= 0 ? dataUrl.slice(0, comma) : '';
  const base64 = comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
  const mimeMatch = header.match(/:(.*?);/);
  const mime = mimeMatch ? mimeMatch[1] : 'image/png';
  const binary = atob(base64);
  const len = binary.length;
  const buffer = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    buffer[i] = binary.charCodeAt(i);
  }
  return new Blob([buffer], { type: mime });
}

/** Decode a `chrome.tabs.captureVisibleTab` data URL without touching the DOM or network stack. */
export async function dataUrlToBitmap(dataUrl: string): Promise<ImageBitmap> {
  const blob = dataUrlToBlob(dataUrl);
  return createImageBitmap(blob);
}

/**
 * Shrink + re-encode a capture before it is handed to the local model.
 *
 * `captureVisibleTab` returns a PNG at devicePixelRatio, which on a HiDPI
 * display is several megabytes of base64. That string has to cross
 * `chrome.runtime.sendMessage` to the offscreen document, be decoded, and then
 * be resampled again by the image processor — slow enough to blow the worker's
 * inference timeout. One downscale here fixes all three costs.
 *
 * This stays unredacted on purpose: it feeds the on-device model only. The
 * escalation path uses `redactFrame` instead.
 */
export async function downscaleFrame(
  dataUrl: string,
  maxWidth = DEFAULTS.maxFrameWidth,
  quality = 0.85,
): Promise<{ dataUrl: string; width: number; height: number }> {
  const bitmap = await dataUrlToBitmap(dataUrl);
  try {
    const scale = Math.min(1, maxWidth / Math.max(1, bitmap.width));
    const w = Math.max(1, Math.round(bitmap.width * scale));
    const h = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return { dataUrl, width: bitmap.width, height: bitmap.height };
    ctx.drawImage(bitmap, 0, 0, w, h);
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
    return { dataUrl: `data:image/jpeg;base64,${await blobToBase64(blob)}`, width: w, height: h };
  } finally {
    bitmap.close();
  }
}

export async function blobToBase64(blob: Blob): Promise<string> {
  const buf = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < buf.length; i += CHUNK) {
    binary += String.fromCharCode(...buf.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function dimensions(s: Source): { width: number; height: number } {
  return { width: (s as ImageBitmap).width ?? 0, height: (s as ImageBitmap).height ?? 0 };
}

function toDrawable(s: Source): CanvasImageSource {
  if (typeof ImageData !== 'undefined' && s instanceof ImageData) {
    const c = new OffscreenCanvas(s.width, s.height);
    c.getContext('2d')?.putImageData(s, 0, 0);
    return c;
  }
  return s as CanvasImageSource;
}

/** Best-effort CSS width of the capture, used when no scale is provided. */
function cssWidthHint(srcW: number): number {
  const dpr = (globalThis as { devicePixelRatio?: number }).devicePixelRatio ?? 1;
  return srcW / (dpr || 1);
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/** Merge contained boxes so the redactor paints fewer, larger rects. */
export function dedupeBoxes(boxes: BoundingBox[]): BoundingBox[] {
  const out: BoundingBox[] = [];
  for (const b of boxes) {
    if (b.width <= 0 || b.height <= 0) continue;
    if (out.some((o) => contains(o, b))) continue;
    const swallowed = out.findIndex((o) => contains(b, o));
    if (swallowed >= 0) out[swallowed] = b;
    else out.push(b);
  }
  return out;
}

const contains = (a: BoundingBox, b: BoundingBox): boolean =>
  a.x <= b.x && a.y <= b.y && a.x + a.width >= b.x + b.width && a.y + a.height >= b.y + b.height;
