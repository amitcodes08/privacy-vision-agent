/**
 * Offscreen host: bridges chrome.runtime messages to the WebGPU worker.
 *
 * The service worker owns orchestration but cannot own a GPU context, and
 * ImageBitmaps cannot cross chrome.runtime — so frames arrive here as
 * base64 data URLs and are decoded locally before transfer to the worker.
 */
import type { WorkerRequest, WorkerResponse } from '~/ai/vlm-worker';
import { newId, type AgentAction, type AgentDecision, type ScrubbedDom } from '@shared/types';

const worker = new Worker(new URL('../ai/vlm-worker.ts', import.meta.url), { type: 'module' });

type Waiter = { resolve: (r: WorkerResponse) => void; reject: (e: Error) => void; want: WorkerResponse['type'] };
const waiters = new Map<string, Waiter>();

worker.addEventListener('message', (ev: MessageEvent<WorkerResponse>) => {
  const msg = ev.data;
  if (msg.type === 'PROGRESS') {
    void chrome.runtime.sendMessage({ target: 'background', kind: 'MODEL_PROGRESS', ...msg }).catch(() => {});
    return;
  }
  const w = waiters.get(msg.id);
  if (!w) return;
  if (msg.type === 'ERROR') {
    waiters.delete(msg.id);
    w.reject(new Error(msg.message));
    return;
  }
  if (msg.type === w.want) {
    waiters.delete(msg.id);
    w.resolve(msg);
  }
});

function ask<T extends WorkerResponse['type']>(
  req: WorkerRequest,
  want: T,
  transfer: Transferable[] = [],
  timeoutMs = 120_000,
): Promise<Extract<WorkerResponse, { type: T }>> {
  return new Promise((resolve, reject) => {
    waiters.set(req.id, { resolve: resolve as (r: WorkerResponse) => void, reject, want });
    const timer = setTimeout(() => {
      waiters.delete(req.id);
      reject(new Error(`worker ${req.type} timed out`));
    }, timeoutMs);
    const settle = <R>(fn: (v: R) => void) => (v: R) => {
      clearTimeout(timer);
      fn(v);
    };
    const w = waiters.get(req.id)!;
    waiters.set(req.id, { ...w, resolve: settle(w.resolve), reject: settle(w.reject) });
    worker.postMessage(req, transfer);
  });
}

export interface OffscreenInferRequest {
  target: 'offscreen';
  kind: 'PROBE' | 'INIT' | 'INFER';
  modelKey?: string;
  goal?: string;
  dom?: ScrubbedDom;
  /** data: URL of the *unredacted* local frame. Never forwarded onward. */
  frameDataUrl?: string;
  history?: AgentAction[];
}

chrome.runtime.onMessage.addListener((message: OffscreenInferRequest, _sender, sendResponse) => {
  if (message?.target !== 'offscreen') return false;
  void (async () => {
    try {
      switch (message.kind) {
        case 'PROBE': {
          const r = await ask({ type: 'PROBE', id: newId() }, 'PROBE_RESULT', [], 10_000);
          sendResponse({ ok: true, webgpu: r.webgpu, adapter: r.adapter, reason: r.reason });
          return;
        }
        case 'INIT': {
          const r = await ask({ type: 'INIT', id: newId(), modelKey: message.modelKey }, 'READY', [], 600_000);
          sendResponse({ ok: true, modelId: r.modelId, device: r.device });
          return;
        }
        case 'INFER': {
          if (!message.frameDataUrl || !message.dom || !message.goal) throw new Error('INFER missing fields');
          const bitmap = await createImageBitmap(await (await fetch(message.frameDataUrl)).blob());
          const r = await ask(
            {
              type: 'INFER',
              id: newId(),
              goal: message.goal,
              dom: message.dom,
              frame: bitmap,
              history: message.history ?? [],
            },
            'DECISION',
            [bitmap],
            45_000,
          );
          sendResponse({ ok: true, decision: r.decision satisfies AgentDecision, raw: r.raw });
          return;
        }
      }
    } catch (err) {
      sendResponse({ ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  })();
  return true; // async response
});
