/// <reference lib="webworker" />
/**
 * Local inference worker. Owns the WebGPU context and the model weights so
 * a long generate() never blocks the offscreen document's message pump.
 *
 * Protocol: postMessage({type, ...}) in, postMessage({type, ...}) out.
 * Everything here is unredacted — this code path must never touch the
 * network beyond the model download.
 */
import { RawImage } from '@huggingface/transformers';
import type { AgentAction, AgentDecision, ScrubbedDom } from '@shared/types';
import {
  DEFAULT_MODEL_KEY,
  MODEL_REGISTRY,
  generateDecision,
  loadModel,
  probeWebGpu,
  type LoadedModel,
} from './model-loader';

export type WorkerRequest =
  | { type: 'PROBE'; id: string }
  | { type: 'INIT'; id: string; modelKey?: string }
  | { type: 'INFER'; id: string; goal: string; dom: ScrubbedDom; frame: ImageBitmap; history?: AgentAction[] }
  | { type: 'DISPOSE'; id: string };

export type WorkerResponse =
  | { type: 'PROBE_RESULT'; id: string; webgpu: boolean; adapter?: string; reason?: string }
  | { type: 'PROGRESS'; id: string; status: string; file?: string; progress?: number; loaded?: number; total?: number }
  | { type: 'READY'; id: string; modelId: string; device: string }
  | { type: 'DECISION'; id: string; decision: AgentDecision; raw: string }
  | { type: 'ERROR'; id: string; message: string };

let loaded: LoadedModel | null = null;
let loading: Promise<LoadedModel> | null = null;

const post = (msg: WorkerResponse) => self.postMessage(msg);

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  void handle(event.data);
});

async function handle(req: WorkerRequest): Promise<void> {
  try {
    switch (req.type) {
      case 'PROBE': {
        const report = await probeWebGpu();
        post({
          type: 'PROBE_RESULT',
          id: req.id,
          webgpu: report.available,
          adapter: report.adapter,
          reason: report.reason,
        });
        return;
      }
      case 'INIT': {
        const key = req.modelKey && MODEL_REGISTRY[req.modelKey] ? req.modelKey : DEFAULT_MODEL_KEY;
        try {
          loading ??= loadModel(key, (p) =>
            post({
              type: 'PROGRESS',
              id: req.id,
              status: p.status,
              file: p.file,
              progress: p.progress,
              loaded: p.loaded,
              total: p.total,
            }),
          );
          loaded = await loading;
          post({ type: 'READY', id: req.id, modelId: loaded.spec.id, device: loaded.device });
        } catch (err) {
          loading = null;
          console.error('[vlm-worker] model loading failed:', err);
          throw err;
        }
        return;
      }
      case 'INFER': {
        // A load already in flight is worth waiting for — bailing out here is
        // what sent the very first steps of every run to the server.
        if (!loaded && loading) {
          loaded = await loading.catch(() => null);
        }
        if (!loaded) {
          // No model: report it as such so the orchestrator falls back to the
          // local planner instead of treating this as a real "escalate".
          post({
            type: 'DECISION',
            id: req.id,
            decision: {
              action: { action: 'escalate', reason: 'local model not initialised' },
              confidence: 0,
              source: 'local',
            },
            raw: '',
          });
          req.frame.close();
          return;
        }
        const image = await bitmapToRawImage(req.frame);
        req.frame.close();
        const result = await generateDecision({
          loaded,
          image,
          goal: req.goal,
          dom: req.dom,
          history: req.history ?? [],
        });
        const { raw, ...decision } = result;
        post({ type: 'DECISION', id: req.id, decision, raw });
        return;
      }
      case 'DISPOSE': {
        await loaded?.model.dispose?.();
        loaded = null;
        loading = null;
        return;
      }
    }
  } catch (err) {
    post({ type: 'ERROR', id: (req as { id: string }).id, message: err instanceof Error ? err.message : String(err) });
  }
}

/** ImageBitmap -> RawImage without a DOM canvas. */
async function bitmapToRawImage(bitmap: ImageBitmap): Promise<RawImage> {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('offscreen 2d context unavailable');
  ctx.drawImage(bitmap, 0, 0);
  const { data, width, height } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  return new RawImage(new Uint8ClampedArray(data), width, height, 4);
}
