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

/**
 * Escalation circuit breaker. With the server down, every step paid a redaction
 * pass plus `waitForOpen`'s 3s timeout and then fell back on-device anyway —
 * while the socket's own retry ladder ran in parallel. Two failures is enough
 * evidence for one run.
 */
const MAX_ESCALATION_FAILURES = 2;
let escalationFailures = 0;
let escalationDown = false;

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

/**
 * `createDocument` resolves once the *document* exists, which is not the same as
 * its message listener being registered — the module script still has to load
 * and run. So the first send after creation could lose the race and come back
 * "Could not establish connection. Receiving end does not exist.", which the
 * warm-up then reported as a model failure. The window is widest in dev, where
 * the offscreen page pulls its script over HTTP from the crxjs dev server.
 *
 * Retried only for that error: everything else is a real reply from a listener
 * that did receive the message.
 */
const OFFSCREEN_HANDSHAKE_ATTEMPTS = 8;
const OFFSCREEN_HANDSHAKE_DELAY_MS = 250;

const isListenerMissing = (msg: string): boolean =>
  /Receiving end does not exist|Could not establish connection|message port closed/i.test(msg);

async function askOffscreen<T>(msg: Record<string, unknown>): Promise<OffscreenReply<T>> {
  let last = 'offscreen document never answered';
  for (let attempt = 1; attempt <= OFFSCREEN_HANDSHAKE_ATTEMPTS; attempt++) {
    try {
      await ensureOffscreen();
      return (await chrome.runtime.sendMessage({ target: 'offscreen', ...msg })) as OffscreenReply<T>;
    } catch (err) {
      last = err instanceof Error ? err.message : String(err);
      if (!isListenerMissing(last)) return { ok: false, error: last };
      // The document may have been torn down between attempts; re-probe it.
      offscreenReady = null;
      await sleep(OFFSCREEN_HANDSHAKE_DELAY_MS);
    }
  }
  return { ok: false, error: `${last} (offscreen listener not ready after ${OFFSCREEN_HANDSHAKE_ATTEMPTS} attempts)` };
}

/**
 * Download weights and create the GPU sessions. Concurrent callers share one
 * attempt.
 *
 * A hard failure is remembered. Without that, `runAgent` re-entered the load on
 * every run and re-ran a download that had already failed for a structural
 * reason, logging the same multi-line error each time. `force` — the popup's
 * WARM_UP — clears it, because an explicit retry is a user asking for exactly
 * that.
 */
let warmUp: Promise<boolean> | null = null;
let warmUpFailed = false;

function warmUpLocalModel(settings: Settings, opts: { force?: boolean } = {}): Promise<boolean> {
  if (warmUpFailed && !opts.force) return Promise.resolve(false);
  if (opts.force) warmUpFailed = false;
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
        status.modelError = undefined;
        logger.info('local', `model ready on ${init.device}`, init.modelId);
        return true;
      }
      modelReady = false;
      warmUpFailed = true;
      status.localModelReady = false;
      status.modelStage = 'load failed';
      // Put the cause in the message, not the data payload: the popup renders
      // messages, so "model init failed" alone was unactionable.
      status.modelError = init.error;
      logger.warn('local', `model init failed (${init.error}); using the on-device planner instead`);
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
 * Frame capture
 * ---------------------------------------------------------------- */

/**
 * Chrome caps `captureVisibleTab` at MAX_CAPTURE_VISIBLE_TAB_CALLS_PER_SECOND
 * (2/s) and rejects the call over quota. A step can complete in well under
 * 500ms when the deterministic planner answers, so the loop tripped the quota
 * and — because the rejection propagated out of `step()` — aborted the entire
 * run. Two rules fix it: never capture faster than the quota allows, and never
 * let a capture failure end the run.
 */
const CAPTURE_MIN_INTERVAL_MS = 600;
let lastCaptureAt = 0;
let captureChain: Promise<unknown> = Promise.resolve();

async function captureFrame(): Promise<string> {
  // Serialise: two concurrent captures would both read a stale `lastCaptureAt`.
  const run = captureChain.then(async () => {
    const wait = CAPTURE_MIN_INTERVAL_MS - (Date.now() - lastCaptureAt);
    if (wait > 0) await sleep(wait);
    try {
      return await chrome.tabs.captureVisibleTab({ format: 'jpeg', quality: 90 });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (!/MAX_CAPTURE_VISIBLE_TAB_CALLS_PER_SECOND/i.test(msg)) throw err;
      // Quota is a per-second window, so one wait is always enough.
      await sleep(CAPTURE_MIN_INTERVAL_MS);
      return await chrome.tabs.captureVisibleTab({ format: 'jpeg', quality: 90 });
    } finally {
      lastCaptureAt = Date.now();
    }
  });
  captureChain = run.catch(() => undefined);
  return run;
}

/**
 * Capture at most once per step, and only if something actually wants pixels.
 *
 * The deterministic planner reads the scrubbed DOM and never looks at the
 * screenshot, so on a run with no local model every one of those captures was
 * both wasted and the reason the quota blew.
 */
function frameOnce(): () => Promise<string> {
  let cached: Promise<string> | null = null;
  return () => (cached ??= captureFrame());
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
  const frame = frameOnce();

  // --- Path 1: local VLM, unredacted, zero network ----------------
  let decision: AgentDecision | null = null;
  if (modelReady) {
    // Downscale once here so the multi-MB HiDPI capture never crosses the
    // runtime message channel or the image processor at full size.
    const rawFrame = await frame();
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
      logger.warn('local', `inference error: ${local.error}`);
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
  const onDevice = () => pickActionable(decision, planned) ?? pickActionable(planLocally({ goal, dom, history }), null);

  // --- Path 3: redact, then escalate -----------------------------
  // `escalationsOff` covers both the setting and the circuit breaker: once the
  // server has failed twice in a run it is not coming back, and every further
  // attempt costs a redaction pass plus a 3s connect timeout per step for
  // nothing.
  if (!settings.allowEscalation || escalationDown) {
    const fallback = onDevice();
    if (fallback) {
      if (fallback.source === 'heuristic') status.heuristicDecisions++;
      else status.localDecisions++;
      logger.info('escalate', `${escalationDown ? 'server unreachable' : 'disabled'}; acting on the best on-device decision`);
      return applyDecision(tabId, fallback);
    }
    logger.warn('escalate', 'unavailable and nothing actionable on-device; stopping');
    return { action: 'done', summary: 'low confidence and no escalation available' };
  }

  const rawFrame = await frame();
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
  try {
    // No explicit connect(): `infer` connects on demand. Calling it here on
    // every step is what spawned a second socket mid-backoff and turned one
    // failed connection into a reconnect storm.
    const response = await ws.infer({
      goal,
      imageBase64: redacted.base64,
      imageMime: redacted.mime,
      dom,
      history,
      localConfidence: best?.confidence ?? 0,
      localReason: (best && reasonOf(best)) ?? 'below threshold',
    });
    escalationFailures = 0;
    status.escalations++;
    logger.info('escalate', `cloud decision ${response.decision.action.action}`, response.rationale);
    return applyDecision(tabId, response.decision);
  } catch (err) {
    // The server is a convenience, not a dependency. Fall back on-device.
    escalationFailures++;
    if (escalationFailures >= MAX_ESCALATION_FAILURES) {
      escalationDown = true;
      ws.close();
      logger.warn('escalate', `${escalationFailures} failures; staying on-device for the rest of this run`);
    }
    logger.warn('escalate', `failed: ${err instanceof Error ? err.message : String(err)}; falling back on-device`);
    const fallback = onDevice();
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
  escalationFailures = 0;
  escalationDown = false;
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
    // Nothing wants the socket between runs, and leaving it open meant its
    // reconnect ladder kept firing long after the agent had stopped.
    ws?.close();
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
          void warmUpLocalModel(settings, { force: true });
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
