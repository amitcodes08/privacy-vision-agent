/**
 * Service worker: the agent orchestrator.
 *
 * Invariant enforced here — the unredacted frame goes only to the offscreen
 * WebGPU worker. The only bytes that reach `WsClient` come out of
 * `redactFrame()`, and only when neither on-device planner could decide.
 *
 * Decision ladder, in order. The whole point is that step 3 is rare:
 *   1. local VLM  (WebGPU, unredacted frame, no network)
 *   2. local deterministic planner  (no model, no network)
 *   3. redacted escalation to the server  (last resort, opt-out-able)
 */
import {
  DEFAULTS,
  type AgentAction,
  type AgentDecision,
  type AgentStatus,
  type BoundingBox,
  type ScrubbedDom,
} from '@shared/types';
import { redactFrame, dataUrlToBitmap, downscaleFrame } from '~/privacy/canvas-redactor';
import { WsClient } from '~/network/ws-client';
import { planLocally } from '~/ai/local-planner';
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
  modelLoading: false,
  escalations: 0,
  localDecisions: 0,
  heuristicDecisions: 0,
  redactions: 0,
};

/* ---------------------------------------------------------------- *
 * Offscreen document lifecycle
 * ---------------------------------------------------------------- */

/**
 * Single-flight. Two concurrent callers (a popup warm-up racing the agent
 * loop, or PROBE racing INIT) would both see zero contexts and both call
 * `createDocument`; the loser throws "Only a single offscreen document may be
 * created", which aborted warm-up and left every step escalating.
 */
let offscreenReady: Promise<void> | null = null;

function ensureOffscreen(): Promise<void> {
  offscreenReady ??= (async () => {
    const existing = await chrome.runtime.getContexts?.({
      contextTypes: ['OFFSCREEN_DOCUMENT' as chrome.runtime.ContextType],
    });
    if (existing && existing.length > 0) return;
    try {
      await chrome.offscreen.createDocument({
        url: OFFSCREEN_PATH,
        reasons: ['WORKERS' as chrome.offscreen.Reason],
        justification: 'Hosts the WebGPU worker that runs the local vision model.',
      });
      logger.info('offscreen', 'document created');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // A parallel creation already won the race — that is success for us.
      if (!/single offscreen document|already exists/i.test(msg)) {
        offscreenReady = null;
        throw err;
      }
    }
  })();
  return offscreenReady;
}

type OffscreenReply<T> = ({ ok: true } & T) | { ok: false; error: string };

async function askOffscreen<T>(msg: Record<string, unknown>): Promise<OffscreenReply<T>> {
  try {
    await ensureOffscreen();
    return (await chrome.runtime.sendMessage({ target: 'offscreen', ...msg })) as OffscreenReply<T>;
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * Download weights and create the GPU sessions. Safe to call repeatedly:
 * concurrent callers share one attempt, and a previous failure does not
 * permanently disable the local path.
 */
let warmUp: Promise<boolean> | null = null;

function warmUpLocalModel(settings: Settings): Promise<boolean> {
  warmUp ??= (async () => {
    status.modelLoading = true;
    status.modelStage = 'probing WebGPU';
    try {
      const probe = await askOffscreen<{ webgpu: boolean; adapter?: string; reason?: string }>({ kind: 'PROBE' });
      webgpuAvailable = probe.ok ? probe.webgpu : false;
      status.webgpuAvailable = webgpuAvailable;
      logger.info('local', `webgpu=${webgpuAvailable}`, probe.ok ? (probe.adapter ?? probe.reason) : probe.error);

      status.modelStage = 'downloading model';
      const init = await askOffscreen<{ modelId: string; device: string }>({
        kind: 'INIT',
        modelKey: settings.modelKey,
      });
      if (init.ok) {
        modelReady = true;
        localModelId = init.modelId;
        status.localModelReady = true;
        status.localModelId = init.modelId;
        status.modelStage = `ready on ${init.device}`;
        status.modelProgress = 100;
        logger.info('local', `model ready on ${init.device}`, init.modelId);
        return true;
      }
      modelReady = false;
      status.localModelReady = false;
      status.modelStage = 'load failed';
      logger.warn('local', 'model init failed; using the on-device planner instead', init.error);
      return false;
    } finally {
      status.modelLoading = false;
      // Allow a later retry rather than pinning the failure for the SW's life.
      warmUp = null;
    }
  })();
  return warmUp;
}

/* ---------------------------------------------------------------- *
 * Content-script RPC
 * ---------------------------------------------------------------- */

interface ScrapeResult {
  dom: ScrubbedDom;
  boxes: BoundingBox[];
  dpr: number;
}

async function waitForTabReady(tabId: number, maxWaitMs = 8_000): Promise<void> {
  const started = Date.now();
  while (Date.now() - started < maxWaitMs) {
    const tab = await chrome.tabs.get(tabId).catch(() => null);
    if (tab && tab.status === 'complete') break;
    await sleep(200);
  }
}

async function ensureContentScript(tabId: number): Promise<void> {
  try {
    const res = (await chrome.tabs.sendMessage(tabId, { kind: 'PING' })) as { ok?: boolean };
    if (res?.ok) return;
  } catch {
    // Ping failed; attempt programmatic injection
  }

  const manifest = chrome.runtime.getManifest();
  const scriptFiles = manifest.content_scripts?.[0]?.js ?? [];
  if (scriptFiles.length > 0) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: scriptFiles,
      });
      await sleep(300);
      logger.info('scripting', `auto-injected content script into tab ${tabId}`);
    } catch (err) {
      logger.warn('scripting', `auto-inject failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}

async function scrape(tabId: number, maxAttempts = 8): Promise<ScrapeResult> {
  const tab = await chrome.tabs.get(tabId).catch(() => null);
  if (!tab?.url || tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://') || tab.url.startsWith('about:')) {
    throw new Error('Browser internal pages cannot be automated. Open a website (e.g. https://news.ycombinator.com) and try again.');
  }

  // If the page is currently navigating/loading, wait for it to complete
  if (tab.status === 'loading') {
    await waitForTabReady(tabId);
  }

  // Ensure content script is active in this tab (handles tab switches & reloads without page refresh)
  await ensureContentScript(tabId);

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const res = (await chrome.tabs.sendMessage(tabId, { kind: 'SCRAPE' })) as OffscreenReply<ScrapeResult>;
      if (res?.ok) return res;
      if (res && res.ok === false) {
        throw new Error(`Scrape error: ${res.error}`);
      }
    } catch (err) {
      if (attempt === 1 || attempt === 3) {
        await ensureContentScript(tabId);
      }
      if (attempt < maxAttempts) {
        await sleep(350);
        continue;
      }
      logger.warn('scrape', `sendMessage to tab ${tabId} failed after ${maxAttempts} attempts: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  const currentTab = await chrome.tabs.get(tabId).catch(() => tab);
  throw new Error(`Content script not connected to tab. Please reload the webpage (${currentTab?.url?.slice(0, 35) ?? ''}...) and try again.`);
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

/**
 * One observe-decide-act cycle.
 *
 * The local VLM is the planner. The deterministic ranker's job is to make sure
 * the model can see the elements that matter (it picks the prompt's element
 * list) and to corroborate what the model then chooses — both of which happen
 * inside the worker. It only *plans* on its own when there is no working model
 * to plan with, which is the one case where the alternative is nothing at all.
 */
async function step(tabId: number, goal: string, history: AgentAction[], settings: Settings): Promise<AgentAction> {
  const { dom, boxes, dpr } = await scrape(tabId);
  status.redactions += boxes.length;
  const rawFrame = await chrome.tabs.captureVisibleTab({ format: 'jpeg', quality: 90 });

  // --- Path 1: local VLM, unredacted, zero network ----------------
  let decision: AgentDecision | null = null;
  if (modelReady) {
    // Downscale once here so the multi-MB HiDPI capture never crosses the
    // runtime message channel or the image processor at full size.
    const small = await downscaleFrame(rawFrame, DEFAULTS.maxFrameWidth).catch(() => ({ dataUrl: rawFrame }));
    const local = await askOffscreen<{ decision: AgentDecision; raw: string }>({
      kind: 'INFER',
      goal,
      dom,
      frameDataUrl: small.dataUrl,
      history,
    });
    if (local.ok) {
      decision = local.decision;
      logger.info(
        'local',
        `vlm ${decision.action.action} conf=${decision.confidence.toFixed(2)} ${decision.latencyMs ?? '?'}ms`,
        local.raw.slice(0, 200),
      );
    } else {
      logger.warn('local', 'inference error', local.error);
    }
  }

  if (decision && decision.confidence >= settings.confidenceThreshold && decision.action.action !== 'escalate') {
    status.localDecisions++;
    return applyDecision(tabId, decision);
  }

  // --- Path 2: no usable model, so plan on-device -----------------
  // Only when the VLM could not produce a decision at all: not loaded yet,
  // errored, or emitted nothing parseable. A model that *did* choose an element
  // and is merely hesitant is not overridden here — a keyword match is not
  // better evidence than a vision model that read the page, and escalation
  // exists precisely for that case.
  const vlmUnusable = !decision || decision.confidence === 0;
  const planned = vlmUnusable ? planLocally({ goal, dom, history }) : null;
  if (planned) {
    logger.info(
      'planner',
      `no usable vlm output; planner says ${planned.action.action} conf=${planned.confidence.toFixed(2)}`,
      reasonOf(planned),
    );
    if (planned.confidence >= settings.confidenceThreshold && planned.action.action !== 'escalate') {
      status.heuristicDecisions++;
      return applyDecision(tabId, planned);
    }
  }

  const best = decision ?? planned;

  // --- Path 3: redact, then escalate -----------------------------
  if (!settings.allowEscalation) {
    // Escalation is off, so act on the best on-device guess rather than
    // stalling — but only if it names a real element.
    const fallback = pickActionable(decision, planned);
    if (fallback) {
      if (fallback === planned) status.heuristicDecisions++;
      else status.localDecisions++;
      logger.info('escalate', 'disabled; acting on the best on-device decision');
      return applyDecision(tabId, fallback);
    }
    logger.warn('escalate', 'blocked by settings; stopping');
    return { action: 'done', summary: 'low confidence and escalation disabled' };
  }

  const bitmap = await dataUrlToBitmap(rawFrame);
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
  try {
    const response = await ws.infer({
      goal,
      imageBase64: redacted.base64,
      imageMime: redacted.mime,
      dom,
      history,
      localConfidence: best?.confidence ?? 0,
      localReason: (best && reasonOf(best)) ?? 'below threshold',
    });
    status.escalations++;
    logger.info('escalate', `cloud decision ${response.decision.action.action}`, response.rationale);
    return applyDecision(tabId, response.decision);
  } catch (err) {
    // The server is a convenience, not a dependency. Fall back on-device.
    logger.warn('escalate', `failed: ${err instanceof Error ? err.message : String(err)}; falling back on-device`);
    const fallback = pickActionable(decision, planned) ?? pickActionable(planLocally({ goal, dom, history }), null);
    if (fallback) {
      if (fallback.source === 'heuristic') status.heuristicDecisions++;
      else status.localDecisions++;
      return applyDecision(tabId, fallback);
    }
    return { action: 'done', summary: 'no decision available on-device and escalation failed' };
  }
}

const reasonOf = (d: AgentDecision): string | undefined => {
  const a = d.action;
  if ('reason' in a && a.reason) return a.reason;
  if (a.action === 'done') return a.summary;
  return undefined;
};

/**
 * The best decision that would actually do something, preferring the model's
 * over the ranker's. `done`/`escalate` are not actions, so they do not count.
 */
function pickActionable(...candidates: (AgentDecision | null)[]): AgentDecision | null {
  for (const c of candidates) {
    if (!c) continue;
    const kind = c.action.action;
    if (kind !== 'escalate' && kind !== 'done') return c;
  }
  return null;
}

/** Identity of an action for loop detection — kind plus target, no reason text. */
const fingerprint = (a: AgentAction): string =>
  'selector' in a && a.selector ? `${a.action}:${a.selector}` : a.action;

/** How many times the most recent action repeats consecutively at the tail. */
function repeatedTail(history: readonly AgentAction[]): number {
  const last = history.at(-1);
  if (!last) return 0;
  const key = fingerprint(last);
  let n = 0;
  for (let i = history.length - 1; i >= 0 && fingerprint(history[i]!) === key; i--) n++;
  return n;
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
  status.maxSteps = settings.maxSteps;
  status.lastError = undefined;
  const history: AgentAction[] = [];

  try {
    // Kick off (or join) the model load, but do not block the run on it: the
    // on-device planner can already act while weights download.
    if (!modelReady) void warmUpLocalModel(settings);

    for (let i = 0; i < settings.maxSteps && !stopRequested; i++) {
      status.step = i + 1;
      const action = await step(tabId, goal, history, settings);
      history.push(action);
      if (action.action === 'done') break;

      // Stuck-loop guard. Without this, an agent repeating one ineffective
      // action burns the whole step budget — and every one of those steps that
      // fell through to escalation was a wasted upload.
      if (repeatedTail(history) >= 3) {
        logger.warn('agent', `same action ${fingerprint(action)} three times; stopping`);
        break;
      }

      if (action.action === 'wait') await sleep(Math.min(action.ms, 5_000));
      else await sleep(400); // let the page settle before re-observing
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    status.lastError = errorMsg;
    logger.error('agent', `run aborted: ${errorMsg}`);
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
            (await chrome.tabs.query({ active: true, lastFocusedWindow: true }))[0]?.id ??
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
        case 'MODEL_PROGRESS': {
          const pct = typeof msg.progress === 'number' ? Math.round(msg.progress) : undefined;
          if (pct !== undefined) status.modelProgress = pct;
          if (typeof msg.status === 'string') status.modelStage = msg.status;
          sendResponse({ ok: true });
          return;
        }
        case 'WARM_UP': {
          const settings = await loadSettings();
          // Respond immediately; a multi-hundred-MB download must not sit on an
          // open message port. The popup polls `status` for progress.
          sendResponse({ ok: true, status });
          void warmUpLocalModel(settings);
          return;
        }
        default:
          sendResponse({ ok: false, error: `unknown kind ${String(msg.kind)}` });
      }
    } catch (err) {
      sendResponse({ ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  })();
  return true;
});

/** Start warming the local model as soon as the worker spins up. */
async function autoWarmUp(): Promise<void> {
  const settings = await loadSettings();
  if (!settings.autoLoadModel) return;
  logger.info('local', 'auto warm-up');
  await warmUpLocalModel(settings);
}

chrome.runtime.onInstalled.addListener(() => {
  logger.info('sw', 'installed');
  void autoWarmUp();
});
chrome.runtime.onStartup.addListener(() => void autoWarmUp());
