/**
 * Service worker: the agent orchestrator.
 *
 * Invariant enforced here — the unredacted frame goes only to the offscreen
 * WebGPU worker. The only bytes that reach `WsClient` come out of
 * `redactFrame()`, and only when the local model's confidence is too low.
 */
import {
  DEFAULTS,
  type AgentAction,
  type AgentDecision,
  type AgentStatus,
  type BoundingBox,
  type ScrubbedDom,
} from '@shared/types';
import { redactFrame, dataUrlToBitmap } from '~/privacy/canvas-redactor';
import { WsClient } from '~/network/ws-client';
import { loadSettings, type Settings } from '~/lib/settings';
import { logger, recentLogs } from '~/lib/log';

const OFFSCREEN_PATH = 'src/offscreen/index.html';

let ws: WsClient | null = null;
let running = false;
let stopRequested = false;
let modelReady = false;
let webgpuAvailable = false;
let localModelId: string | undefined;

const status: AgentStatus = {
  running: false,
  step: 0,
  wsConnected: false,
  webgpuAvailable: false,
  localModelReady: false,
  escalations: 0,
  localDecisions: 0,
  redactions: 0,
};

/* ---------------------------------------------------------------- *
 * Offscreen document lifecycle
 * ---------------------------------------------------------------- */

async function ensureOffscreen(): Promise<void> {
  const existing = await chrome.runtime.getContexts?.({
    contextTypes: ['OFFSCREEN_DOCUMENT' as chrome.runtime.ContextType],
  });
  if (existing && existing.length > 0) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_PATH,
    reasons: ['WORKERS' as chrome.offscreen.Reason],
    justification: 'Hosts the WebGPU worker that runs the local vision model.',
  });
  logger.info('offscreen', 'document created');
}

type OffscreenReply<T> = ({ ok: true } & T) | { ok: false; error: string };

async function askOffscreen<T>(msg: Record<string, unknown>): Promise<OffscreenReply<T>> {
  await ensureOffscreen();
  return (await chrome.runtime.sendMessage({ target: 'offscreen', ...msg })) as OffscreenReply<T>;
}

async function warmUpLocalModel(settings: Settings): Promise<void> {
  const probe = await askOffscreen<{ webgpu: boolean; adapter?: string; reason?: string }>({ kind: 'PROBE' });
  webgpuAvailable = probe.ok ? probe.webgpu : false;
  status.webgpuAvailable = webgpuAvailable;
  logger.info('local', `webgpu=${webgpuAvailable}`, probe.ok ? probe.adapter ?? probe.reason : probe.error);

  const init = await askOffscreen<{ modelId: string; device: string }>({ kind: 'INIT', modelKey: settings.modelKey });
  if (init.ok) {
    modelReady = true;
    localModelId = init.modelId;
    status.localModelReady = true;
    status.localModelId = init.modelId;
    logger.info('local', `model ready on ${init.device}`, init.modelId);
  } else {
    modelReady = false;
    status.localModelReady = false;
    logger.warn('local', 'model init failed; all steps will escalate', init.error);
  }
}

/* ---------------------------------------------------------------- *
 * Content-script RPC
 * ---------------------------------------------------------------- */

interface ScrapeResult {
  dom: ScrubbedDom;
  boxes: BoundingBox[];
  dpr: number;
}

async function scrape(tabId: number): Promise<ScrapeResult> {
  const res = (await chrome.tabs.sendMessage(tabId, { kind: 'SCRAPE' })) as OffscreenReply<ScrapeResult>;
  if (!res?.ok) throw new Error(`scrape failed: ${res?.ok === false ? res.error : 'no response'}`);
  return res;
}

async function execute(tabId: number, action: AgentAction): Promise<{ ok: boolean; error?: string }> {
  const res = (await chrome.tabs.sendMessage(tabId, { kind: 'EXECUTE', action })) as OffscreenReply<{
    detail?: string;
  }>;
  return res?.ok ? { ok: true } : { ok: false, error: res?.ok === false ? res.error : 'no response' };
}

/* ---------------------------------------------------------------- *
 * The loop
 * ---------------------------------------------------------------- */

async function step(tabId: number, goal: string, history: AgentAction[], settings: Settings): Promise<AgentAction> {
  const { dom, boxes, dpr } = await scrape(tabId);
  status.redactions += boxes.length;
  const frameDataUrl = await chrome.tabs.captureVisibleTab({ format: 'png' });

  // --- Path A: local, unredacted, zero network -------------------
  let decision: AgentDecision = {
    action: { action: 'escalate', reason: 'local model unavailable' },
    confidence: 0,
    source: 'local',
  };
  if (modelReady) {
    const local = await askOffscreen<{ decision: AgentDecision; raw: string }>({
      kind: 'INFER',
      goal,
      dom,
      frameDataUrl,
      history,
    });
    if (local.ok) {
      decision = local.decision;
      logger.info('local', `decision ${decision.action.action} conf=${decision.confidence.toFixed(2)}`, local.raw);
    } else {
      logger.warn('local', 'inference error', local.error);
    }
  }

  if (decision.confidence >= settings.confidenceThreshold && decision.action.action !== 'escalate') {
    status.localDecisions++;
    return applyDecision(tabId, decision);
  }

  // --- Path B: redact, then escalate ----------------------------
  if (!settings.allowEscalation) {
    logger.warn('escalate', 'blocked by settings; stopping');
    return { action: 'done', summary: 'low confidence and escalation disabled' };
  }

  const bitmap = await dataUrlToBitmap(frameDataUrl);
  const redacted = await redactFrame(bitmap, {
    boxes,
    scale: dpr,
    style: settings.redactionStyle,
    maxWidth: DEFAULTS.maxFrameWidth,
    quality: DEFAULTS.jpegQuality,
  });
  bitmap.close();
  logger.info(
    'redact',
    `${redacted.boxesApplied}/${boxes.length} boxes, ${Math.round(redacted.bytes / 1024)}KB, ${redacted.elapsedMs.toFixed(1)}ms`,
  );

  ws ??= makeWsClient(settings);
  ws.connect();
  const response = await ws.infer({
    goal,
    imageBase64: redacted.base64,
    imageMime: redacted.mime,
    dom,
    history,
    localConfidence: decision.confidence,
    localReason: decision.action.action === 'escalate' ? decision.action.reason : 'below threshold',
  });
  status.escalations++;
  logger.info('escalate', `cloud decision ${response.decision.action.action}`, response.rationale);
  return applyDecision(tabId, response.decision);
}

async function applyDecision(tabId: number, decision: AgentDecision): Promise<AgentAction> {
  status.lastDecision = decision;
  const action = decision.action;
  if (action.action === 'done' || action.action === 'escalate') return action;
  const res = await execute(tabId, action);
  if (!res.ok) logger.warn('execute', `${action.action} failed`, res.error);
  return action;
}

export async function runAgent(goal: string, tabId: number): Promise<void> {
  if (running) throw new Error('agent already running');
  const settings = await loadSettings();
  running = true;
  stopRequested = false;
  status.running = true;
  status.goal = goal;
  status.step = 0;
  const history: AgentAction[] = [];

  try {
    if (!modelReady) await warmUpLocalModel(settings);
    for (let i = 0; i < settings.maxSteps && !stopRequested; i++) {
      status.step = i + 1;
      const action = await step(tabId, goal, history, settings);
      history.push(action);
      if (action.action === 'done') break;
      if (action.action === 'wait') await sleep(Math.min(action.ms, 5_000));
      else await sleep(400); // let the page settle before re-observing
    }
  } catch (err) {
    logger.error('agent', 'run aborted', err instanceof Error ? err.message : err);
  } finally {
    running = false;
    status.running = false;
  }
}

function makeWsClient(settings: Settings): WsClient {
  return new WsClient({
    url: settings.wsUrl,
    extensionVersion: chrome.runtime.getManifest().version,
    webgpu: webgpuAvailable,
    localModelId,
    onState: (s) => {
      status.wsConnected = s === 'open';
      logger.info('ws', `state ${s}`);
    },
    onError: (e) => logger.warn('ws', `${e.code}: ${e.message}`),
    onLog: (m, d) => logger.debug('ws', m, d),
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/* ---------------------------------------------------------------- *
 * Wiring
 * ---------------------------------------------------------------- */

chrome.runtime.onMessage.addListener((msg: Record<string, unknown>, _sender, sendResponse) => {
  if (msg?.target === 'offscreen') return false; // not ours
  void (async () => {
    try {
      switch (msg.kind) {
        case 'AGENT_START': {
          const tabId =
            (msg.tabId as number | undefined) ??
            (await chrome.tabs.query({ active: true, currentWindow: true }))[0]?.id;
          if (!tabId) throw new Error('no active tab');
          sendResponse({ ok: true, started: true });
          void runAgent(String(msg.goal ?? ''), tabId);
          return;
        }
        case 'AGENT_STOP':
          stopRequested = true;
          sendResponse({ ok: true });
          return;
        case 'AGENT_STATUS_REQUEST':
          sendResponse({ ok: true, status, logs: recentLogs(40) });
          return;
        case 'MODEL_PROGRESS':
          logger.debug('local', String(msg.status ?? 'progress'), msg.progress);
          sendResponse({ ok: true });
          return;
        case 'WARM_UP':
          await warmUpLocalModel(await loadSettings());
          sendResponse({ ok: true, status });
          return;
        default:
          sendResponse({ ok: false, error: `unknown kind ${String(msg.kind)}` });
      }
    } catch (err) {
      sendResponse({ ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  })();
  return true;
});

chrome.runtime.onInstalled.addListener(() => logger.info('sw', 'installed'));
