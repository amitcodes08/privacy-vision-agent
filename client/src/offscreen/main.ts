/**
 * Offscreen host: bridges chrome.runtime messages to the WebGPU worker, and
 * hosts Gemini Nano.
 *
 * The service worker owns orchestration but cannot own a GPU context, and
 * ImageBitmaps cannot cross chrome.runtime — so frames arrive here as
 * base64 data URLs and are decoded locally before transfer to the worker.
 *
 * This document is also the only place in the extension where Chrome's built-in
 * Prompt API exists: it is exposed to documents, not to workers, so the
 * service worker's sub-query calls arrive here as `NANO_*` messages. Note the
 * asymmetry with the VLM — Nano runs on *this* thread rather than in the worker,
 * because the API is not exposed to web workers either.
 */
import type { WorkerRequest, WorkerResponse } from '~/ai/vlm-worker';
import type { SttWorkerRequest, SttWorkerResponse } from '~/ai/stt-worker';
import { dataUrlToBitmap } from '~/privacy/canvas-redactor';
import { newId, type AgentAction, type AgentDecision, type ScrubbedDom } from '@shared/types';
import type { NanoMessageKind } from '~/ai/nano-bridge';
import {
  decomposeGoal,
  nanoPlanner,
  nanoStatus,
  replanFromPage,
  verifySubObjective,
  type ReplanInput,
  type VerifyInput,
} from '~/ai/nano-query-planner';
import { probeNano } from '~/ai/nano-session';

const worker = new Worker(new URL('../ai/vlm-worker.ts', import.meta.url), { type: 'module' });
const sttWorker = new Worker(new URL('../ai/stt-worker.ts', import.meta.url), { type: 'module' });

type Waiter = { resolve: (r: WorkerResponse) => void; reject: (e: Error) => void; want: WorkerResponse['type'] };
const waiters = new Map<string, Waiter>();

type SttWaiter = { resolve: (r: SttWorkerResponse) => void; reject: (e: Error) => void; want: SttWorkerResponse['type'] };
const sttWaiters = new Map<string, SttWaiter>();

/**
 * Per-file download progress, aggregated.
 *
 * `progress_callback` fires per ONNX graph (embed_tokens, vision_encoder,
 * decoder_model_merged) with a 0-100 value for that file alone, so forwarding
 * it raw made the popup's percentage jump backwards. Averaging by bytes gives
 * one number that only moves forward.
 */
const fileProgress = new Map<string, { loaded: number; total: number }>();
let lastPosted = -1;

function forwardProgress(msg: Extract<WorkerResponse, { type: 'PROGRESS' }> & { loaded?: number; total?: number }): void {
  if (msg.file && typeof msg.total === 'number' && msg.total > 0) {
    fileProgress.set(msg.file, { loaded: msg.loaded ?? 0, total: msg.total });
  }

  let percent: number | undefined;
  if (fileProgress.size > 0) {
    let loaded = 0;
    let total = 0;
    for (const f of fileProgress.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    if (total > 0) percent = Math.min(100, Math.round((loaded / total) * 100));
  } else if (typeof msg.progress === 'number') {
    percent = Math.round(msg.progress);
  }

  // Throttle: these fire hundreds of times per file.
  if (percent !== undefined && percent === lastPosted && msg.status === 'progress') return;
  if (percent !== undefined) lastPosted = percent;

  const stage =
    msg.status === 'progress' || msg.status === 'download'
      ? `downloading weights${percent !== undefined ? ` ${percent}%` : ''}`
      : msg.status === 'done'
        ? 'preparing sessions'
        : msg.status;

  void chrome.runtime
    .sendMessage({ target: 'background', kind: 'MODEL_PROGRESS', status: stage, progress: percent })
    .catch(() => {});
}

worker.addEventListener('message', (ev: MessageEvent<WorkerResponse>) => {
  const msg = ev.data;
  if (msg.type === 'PROGRESS') {
    forwardProgress(msg);
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

sttWorker.addEventListener('message', (ev: MessageEvent<SttWorkerResponse>) => {
  const msg = ev.data;
  if (msg.type === 'STT_PROGRESS') {
    const percent = typeof msg.progress === 'number' ? Math.round(msg.progress) : undefined;
    void chrome.runtime
      .sendMessage({ target: 'background', kind: 'STT_PROGRESS', status: msg.status, progress: percent })
      .catch(() => {});
    return;
  }
  const w = sttWaiters.get(msg.id);
  if (!w) return;
  if (msg.type === 'ERROR') {
    sttWaiters.delete(msg.id);
    w.reject(new Error(msg.message));
    return;
  }
  if (msg.type === w.want) {
    sttWaiters.delete(msg.id);
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

function askStt<T extends SttWorkerResponse['type']>(
  req: SttWorkerRequest,
  want: T,
  transfer: Transferable[] = [],
  timeoutMs = 60_000,
): Promise<Extract<SttWorkerResponse, { type: T }>> {
  return new Promise((resolve, reject) => {
    sttWaiters.set(req.id, { resolve: resolve as (r: SttWorkerResponse) => void, reject, want });
    const timer = setTimeout(() => {
      sttWaiters.delete(req.id);
      reject(new Error(`stt-worker ${req.type} timed out`));
    }, timeoutMs);
    const settle = <R>(fn: (v: R) => void) => (v: R) => {
      clearTimeout(timer);
      fn(v);
    };
    const w = sttWaiters.get(req.id)!;
    sttWaiters.set(req.id, { ...w, resolve: settle(w.resolve), reject: settle(w.reject) });
    sttWorker.postMessage(req, transfer);
  });
}

export interface OffscreenInferRequest {
  target: 'offscreen';
  kind: 'PROBE' | 'INIT' | 'INFER' | NanoMessageKind | 'STT_INIT' | 'STT_TRANSCRIBE';
  modelKey?: string;
  goal?: string;
  dom?: ScrubbedDom;
  /** data: URL of the *unredacted* local frame. Never forwarded onward. */
  frameDataUrl?: string;
  history?: AgentAction[];
  taskMemory?: import('@shared/types').TaskMemory;
  /** Payload for NANO_REPLAN / NANO_VERIFY, already JSON-round-tripped. */
  input?: ReplanInput | VerifyInput;
  /** NANO_PROBE only: create a session, downloading weights if Chrome needs to. */
  allowDownload?: boolean;
  /** STT_TRANSCRIBE: base64-encoded Float32Array of 16 kHz mono audio. */
  audioBase64?: string;
  sampleRate?: number;
}

chrome.runtime.onMessage.addListener((message: OffscreenInferRequest, _sender, sendResponse) => {
  if (message?.target !== 'offscreen') return false;
  void (async () => {
    try {
      switch (message.kind) {
        /* ---- Gemini Nano sub-query engine ------------------------------ *
         * These run on the document thread, not in the worker: the Prompt API
         * is exposed to neither the service worker nor web workers. They are
         * cheap (text-only, on-device) so they do not need the worker's
         * isolation — but a slow generation does share this thread with the
         * message pump, which is why every call in `nano-session` is bounded.
         * ---------------------------------------------------------------- */
        case 'NANO_PROBE': {
          // `nanoPlanner` opens (and caches) the session so the first real
          // query does not also pay session creation. With `allowDownload` the
          // caller has a user gesture behind it and Chrome may fetch weights.
          if (message.allowDownload) await nanoPlanner({ allowDownload: true });
          const probe = message.allowDownload ? nanoStatus() : await probeNano();
          sendResponse({ ok: true, probe });
          return;
        }
        case 'NANO_DECOMPOSE': {
          const plan = await decomposeGoal(message.goal ?? '', message.dom);
          sendResponse({ ok: true, plan });
          return;
        }
        case 'NANO_REPLAN': {
          const result = await replanFromPage(message.input as ReplanInput);
          sendResponse({ ok: true, result });
          return;
        }
        case 'NANO_VERIFY': {
          const result = await verifySubObjective(message.input as VerifyInput);
          sendResponse({ ok: true, result });
          return;
        }
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
          const bitmap = await dataUrlToBitmap(message.frameDataUrl);
          const r = await ask(
            {
              type: 'INFER',
              id: newId(),
              goal: message.goal,
              dom: message.dom,
              frame: bitmap,
              history: message.history ?? [],
              taskMemory: message.taskMemory,
            },
            'DECISION',
            [bitmap],
            // The first inference also compiles WebGPU shaders, which can take
            // far longer than steady-state generation.
            90_000,
          );
          sendResponse({ ok: true, decision: r.decision satisfies AgentDecision, raw: r.raw });
          return;
        }
        case 'STT_INIT': {
          await askStt({ type: 'STT_INIT', id: newId() }, 'STT_READY', [], 300_000);
          sendResponse({ ok: true });
          return;
        }
        case 'STT_TRANSCRIBE': {
          if (!message.audioBase64) throw new Error('STT_TRANSCRIBE missing audioBase64');
          const binary = atob(message.audioBase64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          const audio = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
          const r = await askStt(
            { type: 'STT_TRANSCRIBE', id: newId(), audio, sampleRate: message.sampleRate ?? 16_000 },
            'STT_RESULT',
            [audio.buffer],
            60_000,
          );
          sendResponse({ ok: true, transcript: r.transcript });
          return;
        }
      }
    } catch (err) {
      sendResponse({ ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  })();
  return true; // async response
});
