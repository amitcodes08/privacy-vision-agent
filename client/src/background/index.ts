/**
 * Service worker: the agent orchestrator.
 *
 * Invariant enforced here — the unredacted frame goes only to the offscreen
 * WebGPU worker. The only bytes that reach `WsClient` come out of
 * `redactFrame()`, and only when neither on-device planner could decide.
 *
 * Two stages, and they answer different questions.
 *
 * Stage A — *what* to do, via Gemini Nano's sub-query engine. A complex goal is
 * decomposed into atomic sub-objectives before the first action, and rewritten
 * against the real page whenever a sub-objective stalls. Nano is not in the
 * decision ladder below: it never picks an element, it only decides what the
 * current sub-goal is. It lives in the offscreen document, because Chrome
 * exposes the Prompt API to documents and not to workers — see `nano-bridge.ts`.
 *
 * Stage B — *how* to do it, one sub-objective at a time. The whole point is that
 * step 3 is rare:
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
  type TaskMemory,
  type TaskObjective,
  type ActionResultCategory,
} from '@shared/types';
import { redactFrame, dataUrlToBitmap, downscaleFrame } from '~/privacy/canvas-redactor';
import { WsClient } from '~/network/ws-client';
import { planLocally } from '~/ai/local-planner';
import { loadSettings, type Settings } from '~/lib/settings';
import { logger, recentLogs } from '~/lib/log';
import { checkTermination, corroborateDone, HIGH_CONFIDENCE } from '~/ai/termination-checker';
import { makeStagnationState, recordAndCheck, fingerprint } from '~/ai/stagnation-guard';
import { sanitiseCloudAction } from '~/ai/decision-parser';
import { createNanoRouter } from '~/ai/nano-bridge';
import type { NanoProbe } from '~/ai/nano-session';

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
interface StepResult {
  action: AgentAction;
  preDom: ScrubbedDom;
  postDom: ScrubbedDom;
  observed: boolean;
}

async function step(tabId: number, goal: string, history: AgentAction[], settings: Settings, taskMemory: TaskMemory): Promise<StepResult> {
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
      taskMemory,
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
    // --- VLM-done corroboration: verify a done signal against the DOM ------
    if (decision.action.action === 'done') {
      const activeObjective =
        taskMemory.currentObjective?.trim() || goal;

      const domSignal = checkTermination({
        goal: activeObjective,
        dom,
        history,
        taskMemory,
      });

      const corroborated = corroborateDone(
        decision.confidence,
        domSignal,
      );

      if (
        corroborated.done &&
        corroborated.confidence >= HIGH_CONFIDENCE
      ) {
        logger.info(
          'local',
          `vlm done corroborated by objective DOM evidence (${corroborated.reason}); stopping objective`,
        );

        status.localDecisions++;

        return {
          action: decision.action,
          preDom: dom,
          postDom: dom,
          observed: true,
        };
      }

      logger.info(
        'local',
        `rejected vlm done for active objective (${corroborated.reason}); continuing`,
      );

      decision = null;
    } else {
      status.localDecisions++;
      const applied = await applyDecision(tabId, decision, dom);
      return { action: applied.action, preDom: applied.preDom, postDom: applied.postDom, observed: applied.observed };
    }
  }

  // --- Path 2: no usable model, so plan on-device -----------------
  const vlmUnusable = !decision || decision.confidence === 0;
  const planned = vlmUnusable ? planLocally({ goal, dom, history, taskMemory }) : null;
  if (planned) {
    logger.info(
      'planner',
      `no usable vlm output; planner says ${planned.action.action} conf=${planned.confidence.toFixed(2)}`,
      reasonOf(planned),
    );
    if (planned.confidence >= Math.max(settings.confidenceThreshold, 0.60) && planned.action.action !== 'escalate') {
      status.heuristicDecisions++;
      const applied = await applyDecision(tabId, planned, dom);
      return { action: applied.action, preDom: applied.preDom, postDom: applied.postDom, observed: applied.observed };
    }
  }

  const best = decision ?? planned;
  const onDevice = () => pickActionable(decision, planned) ?? pickActionable(planLocally({ goal, dom, history }), null);

  // --- Path 3: redact, then escalate -----------------------------
  if (!settings.allowEscalation || escalationDown) {
    const fallback = onDevice();
    if (fallback) {
      if (fallback.source === 'heuristic') status.heuristicDecisions++;
      else status.localDecisions++;
      logger.info('escalate', `${escalationDown ? 'server unreachable' : 'disabled'}; acting on the best on-device decision`);
      const applied = await applyDecision(tabId, fallback, dom);
      return { action: applied.action, preDom: applied.preDom, postDom: applied.postDom, observed: applied.observed };
    }
    logger.warn('escalate', 'unavailable and nothing actionable on-device; stopping');
    return { action: { action: 'done', summary: 'low confidence and no escalation available' }, preDom: dom, postDom: dom, observed: true };
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
    const response = await ws.infer({
      goal,
      imageBase64: redacted.base64,
      imageMime: redacted.mime,
      dom,
      history,
      taskMemory,
      localConfidence: best?.confidence ?? 0,
      localReason: (best && reasonOf(best)) ?? 'below threshold',
    });
    escalationFailures = 0;
    status.escalations++;
    const cloudDecision = sanitiseCloudAction(response.decision);

    if (cloudDecision.action.action === 'done') {
      const activeObjective =
        taskMemory.currentObjective?.trim() || goal;

      const domSignal = checkTermination({
        goal: activeObjective,
        dom,
        history,
        taskMemory,
      });

      const corroborated = corroborateDone(
        cloudDecision.confidence,
        domSignal,
      );

      if (
        !corroborated.done ||
        corroborated.confidence < HIGH_CONFIDENCE
      ) {
        logger.warn(
          'escalate',
          `cloud done rejected for active objective: ${corroborated.reason}`,
        );

        /*
         * Do not execute a hallucinated DONE.
         * Immediately give the deterministic planner one chance.
         */
        const fallback = planLocally({
          goal,
          dom,
          history,
          taskMemory,
        });

        if (
          fallback.action.action !== 'invalid' &&
          fallback.action.action !== 'done' &&
          fallback.action.action !== 'escalate' &&
          fallback.confidence >= 0.60
        ) {
          const applied = await applyDecision(
            tabId,
            fallback,
            dom,
          );

          return {
            action: applied.action,
            preDom: applied.preDom,
            postDom: applied.postDom,
            observed: applied.observed,
          };
        }

        return {
          action: {
            action: 'invalid',
            reason: 'cloud model claimed completion without objective evidence',
          },
          preDom: dom,
          postDom: dom,
          observed: true,
        };
      }
    }

    logger.info(
      'escalate',
      `cloud decision ${cloudDecision.action.action}`,
      response.rationale,
    );

    const applied = await applyDecision(
      tabId,
      cloudDecision,
      dom,
    );

    return {
      action: applied.action,
      preDom: applied.preDom,
      postDom: applied.postDom,
      observed: applied.observed,
    };
  } catch (err) {
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
      const applied = await applyDecision(tabId, fallback, dom);
      return { action: applied.action, preDom: applied.preDom, postDom: applied.postDom, observed: applied.observed };
    }
    return { action: { action: 'done', summary: 'no decision available on-device and escalation failed' }, preDom: dom, postDom: dom, observed: true };
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
    if (kind !== 'escalate' && kind !== 'done' && kind !== 'invalid') return c;
  }
  return null;
}

/** Identity of an action for loop detection — kind plus target, no reason text. */
const domFingerprint = (a: AgentAction): string =>
  'selector' in a && a.selector ? `${a.action}:${a.selector}` : a.action;

/** How many times the most recent action repeats consecutively at the tail. */
function repeatedTail(history: readonly AgentAction[]): number {
  const last = history.at(-1);
  if (!last) return 0;
  const key = domFingerprint(last);
  let n = 0;
  for (let i = history.length - 1; i >= 0 && domFingerprint(history[i]!) === key; i--) n++;
  return n;
}

async function applyDecision(
  tabId: number,
  decision: AgentDecision,
  initialPreDom: ScrubbedDom,
): Promise<{ action: AgentAction; preDom: ScrubbedDom; postDom: ScrubbedDom; observed: boolean }> {
  status.lastDecision = decision;
  let action = decision.action;
  if (action.action === 'done' || action.action === 'escalate' || action.action === 'invalid') {
    return { action, preDom: initialPreDom, postDom: initialPreDom, observed: true };
  }

  // Safe pre-execution snapshot: avoid unhandled rejection if tab navigated or disconnected during inference
  let immediatePreDom = initialPreDom;
  try {
    const preScrape = await scrape(tabId);
    immediatePreDom = preScrape.dom;
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    logger.warn('execute', `pre-execution scrape failed: ${errorMsg}`);
    return {
      action: { action: 'invalid', reason: errorMsg || 'tab unreachable before execution' },
      preDom: initialPreDom,
      postDom: initialPreDom,
      observed: false,
    };
  }

  // Reconcile stale selectors if DOM updated during inference
  //
  // `staleSelector` is captured rather than read through `action` inside the
  // closures below: `action` is reassigned in this block, so TypeScript drops
  // the `'selector' in action` narrowing for anything a callback captures.
  const staleSelector = 'selector' in action ? action.selector : undefined;
  if (staleSelector && fingerprint(initialPreDom) !== fingerprint(immediatePreDom)) {
    const origNode = initialPreDom.nodes.find((n) => n.selector === staleSelector);
    const targetNode = immediatePreDom.nodes.find((n) => n.selector === staleSelector);

    const isSameElement =
      origNode &&
      targetNode &&
      origNode.tag === targetNode.tag &&
      (origNode.name === targetNode.name || origNode.label === targetNode.label || origNode.text === targetNode.text);

    if (!isSameElement && origNode) {
      const origBox = origNode.box;
      // Find all candidates in the updated DOM matching the original node's key semantic attributes
      const candidates = immediatePreDom.nodes.filter(
        (n) =>
          n.tag === origNode.tag &&
          ((origNode.label && n.label === origNode.label) ||
            (origNode.name && n.name === origNode.name) ||
            (origNode.placeholder && n.placeholder === origNode.placeholder) ||
            (origNode.text && n.text === origNode.text)),
      );

      // Disambiguate by spatial proximity when the original position is known.
      const closest =
        candidates.length > 1 && origBox
          ? candidates
            .slice()
            .sort(
              (a, b) =>
                Math.hypot((a.box?.x ?? 0) - origBox.x, (a.box?.y ?? 0) - origBox.y) -
                Math.hypot((b.box?.x ?? 0) - origBox.x, (b.box?.y ?? 0) - origBox.y),
            )[0]
          : candidates.length === 1
            ? candidates[0]
            : undefined;

      if (closest) {
        const how = candidates.length === 1 ? 'reconciled' : 'disambiguated';
        logger.info('execute', `${how} stale selector ${staleSelector} -> ${closest.selector}`);
        // Re-narrow: `staleSelector` came off this same action, but reading it
        // through a separate const means TS no longer knows that.
        if ('selector' in action) action = { ...action, selector: closest.selector };
      } else {
        logger.warn('execute', `stale selector ${staleSelector} has ${candidates.length} matches in updated DOM; rejecting for safety`);
        return {
          action: { action: 'invalid', reason: `target element ambiguous or vanished after DOM change (${staleSelector})` },
          preDom: immediatePreDom,
          postDom: immediatePreDom,
          observed: true,
        };
      }
    }
  }

  const res = await execute(tabId, action);
  if (!res.ok) {
    logger.warn('execute', `${action.action} failed`, res.error);
    return { action: { action: 'invalid', reason: res.error || 'execution failed' }, preDom: immediatePreDom, postDom: immediatePreDom, observed: true };
  }

  // Adaptive settlement for state-changing actions
  await sleep(100);
  let postDom: ScrubbedDom | undefined;

  if (action.action === 'navigate') {
    await waitForTabReady(tabId);
    await ensureContentScript(tabId);
  }

  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      let post = await scrape(tabId);
      if (action.action === 'click' || action.action === 'fill' || action.action === 'navigate') {
        if (fingerprint(immediatePreDom) === fingerprint(post.dom) && attempt < 3) {
          await sleep(350);
          continue;
        }
      }
      postDom = post.dom;
      break;
    } catch (err) {
      logger.warn('execute', `post-execution scrape attempt ${attempt} failed: ${err instanceof Error ? err.message : String(err)}`);
      if (attempt < 3) {
        await waitForTabReady(tabId, 2000);
        await ensureContentScript(tabId);
        await sleep(200);
      }
    }
  }

  const observed = postDom !== undefined;
  return {
    action,
    preDom: immediatePreDom,
    postDom: postDom ?? immediatePreDom,
    observed,
  };
}


/* ---------------------------------------------------------------- *
 * Stage A: the Gemini Nano sub-query engine
 * ---------------------------------------------------------------- */

/**
 * Nano runs wherever the Prompt API exists — in practice the offscreen
 * document, reached over the same channel as the VLM.
 */
const nano = createNanoRouter((msg) => askOffscreen(msg));

/**
 * Re-plans allowed per run. Each one costs a Nano generation and resets the
 * plan the user is watching, and a goal that needs a third rewrite is a goal
 * this agent is not going to finish.
 */
const MAX_REPLANS = 2;

const activeIndexOf = (tm: TaskMemory): number =>
  tm.subObjectives?.findIndex((s) => s.status === 'active') ?? -1;

/** Mirror the plan into `status` so the popup can render the checklist. */
function publishPlan(tm: TaskMemory): void {
  status.plan = {
    source: tm.planSource ?? 'local-rules',
    objectives: (tm.subObjectives ?? []).map((o) => ({ ...o })),
    replans: tm.replans ?? 0,
  };
}

/**
 * Retire the active sub-objective and activate the next one.
 * Returns false when that was the last, i.e. the whole goal is done.
 */
function advanceObjective(tm: TaskMemory, why: string): boolean {
  const subs = tm.subObjectives;
  const at = activeIndexOf(tm);
  const current = subs?.[at];
  if (!subs || !current) return false;

  current.status = 'completed';
  tm.completedObjectives.push({ ...current });
  tm.attemptedTargets = [];
  logger.info('task', `completed sub-objective [${at + 1}/${subs.length}]: ${current.description}`, why);

  const next = subs[at + 1];
  if (next) {
    next.status = 'active';
    tm.currentObjective = next.description;
    logger.info('task', `advancing to [${at + 2}/${subs.length}]: ${next.description}`);
    publishPlan(tm);
    return true;
  }

  tm.currentObjective = undefined;
  logger.info('task', 'all sub-objectives finished');
  publishPlan(tm);
  return false;
}

/**
 * Did the active sub-objective just land?
 *
 * Three judges, cheapest first: an explicit `completedObjective` from whichever
 * model chose the action, the deterministic DOM checker, then Nano looking at
 * the page. Any one of them is enough to advance.
 *
 * That asymmetry is deliberate. A missed advance strands the agent on a
 * finished sub-goal until the step budget runs out and the run fails; an early
 * advance is self-correcting, because the next sub-objective's own check will
 * not fire and a re-plan gets to look at the real page. Nano is asked last so a
 * generation is skipped entirely whenever the free evidence already answered.
 */
async function subObjectiveSatisfied(
  objective: string,
  action: AgentAction,
  result: StepResult,
  history: readonly AgentAction[],
): Promise<{ done: boolean; why: string }> {
  /*
   * completedObjective is a MODEL CLAIM, not evidence.
   * Never advance solely because the model wrote it.
   */

  const domSignal = checkTermination({
    goal: objective,
    dom: result.postDom,
    lastAction: action,
    prevDom: result.preDom,
    history: [...history],
  });

  if (
    domSignal.done &&
    domSignal.confidence >= HIGH_CONFIDENCE
  ) {
    return {
      done: true,
      why: `dom: ${domSignal.reason}`,
    };
  }

  const verdict = await nano.verify({
    objective,
    dom: result.postDom,
    lastAction: action,
  });

  if (
    verdict.satisfied &&
    verdict.confidence > 0
  ) {
    return {
      done: true,
      why: `nano: ${verdict.reason}`,
    };
  }

  return {
    done: false,
    why:
      verdict.source === 'gemini-nano'
        ? `nano: ${verdict.reason}`
        : domSignal.reason,
  };
}

/**
 * Rewrite the remaining plan against the page actually in front of us.
 *
 * This is the case a decomposition made before the agent saw any page cannot
 * handle. "Filter by price under $500" is a reasonable sub-goal and not a
 * control that exists on most sites; the vision tier then hunts for an element
 * that was never there, the stagnation guard fires, and the run dies having
 * done the first half of the task. Given the page's element list, Nano can
 * re-say the outstanding work in terms the vision tier can ground.
 *
 * Superseded objectives are marked `skipped` rather than dropped, so the popup
 * still shows the plan the run started with and `checkTermination` does not
 * count them as outstanding work.
 */
async function tryReplan(
  tm: TaskMemory,
  dom: ScrubbedDom,
  history: readonly AgentAction[],
  reason: string,
): Promise<boolean> {
  const spent = tm.replans ?? 0;
  if (spent >= MAX_REPLANS) {
    logger.info('plan', `stalled (${reason}) with the re-plan budget spent (${spent}/${MAX_REPLANS})`);
    return false;
  }

  const subs = tm.subObjectives ?? [];
  const at = activeIndexOf(tm);
  const remaining = at === -1 ? [] : subs.slice(at).filter((s) => s.status !== 'completed' && s.status !== 'skipped');
  if (remaining.length === 0) return false;

  const res = await nano.replan({
    goal: tm.goal,
    remaining,
    completed: tm.completedObjectives,
    dom,
    history: [...history],
    reason,
  });
  if (!res.changed || res.subObjectives.length === 0) {
    logger.info('plan', `no usable re-plan for "${reason}"; falling through`);
    return false;
  }

  for (const s of subs.slice(at)) {
    if (s.status !== 'completed') s.status = 'skipped';
  }
  const appended: TaskObjective[] = res.subObjectives.map((o, i) => ({ ...o, id: subs.length + i + 1 }));
  subs.push(...appended);

  tm.subObjectives = subs;
  tm.planSource = res.source;
  tm.replans = spent + 1;
  tm.currentObjective = appended[0]?.description;
  // The old plan's dead ends say nothing about the new plan's targets.
  tm.attemptedTargets = [];
  publishPlan(tm);

  logger.info(
    'plan',
    `re-planned after ${reason} (${tm.replans}/${MAX_REPLANS})`,
    appended.map((o) => o.description).join(' -> '),
  );
  return true;
}

/**
 * Probe Nano and, if Chrome has not downloaded the weights yet, ask it to.
 *
 * Must be reached from a user gesture — see the WARM_UP handler. Failure is
 * recorded in `status.nano` and nothing else: the run path treats a missing Nano
 * as "use the rule-based splitter", not as an error.
 */
async function warmUpNano(): Promise<void> {
  // `allowDownload` opens a session in the offscreen document, which is what
  // makes Chrome fetch the weights; the router's own probe afterwards then sees
  // 'available' rather than 'downloadable'.
  const opened = await askOffscreen<{ probe: NanoProbe }>({ kind: 'NANO_PROBE', allowDownload: true });
  nano.reset();
  const route = await nano.route();
  const probe = nano.status();
  status.nano = { route, state: probe.state, reason: probe.reason };
  logger.info('nano', `warm-up: route=${route} state=${probe.state}`, opened.ok ? probe.reason : opened.error);
}

/**
 * The page the *first* decomposition gets to see.
 *
 * Optional on purpose: a blind decomposition is still useful, and a failed
 * scrape here must not pre-empt the much better error message `step()` produces
 * on the same failure one moment later.
 */
async function seedDom(tabId: number): Promise<ScrubbedDom | undefined> {
  try {
    return (await scrape(tabId, 2)).dom;
  } catch {
    return undefined;
  }
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
  status.plan = undefined;
  const history: AgentAction[] = [];

  try {
    // Stage A. Where Nano runs is resolved once per service-worker lifetime;
    // logging it is the only way to tell "no Nano on this machine" apart from
    // "Nano is there and the plan really is one step".
    const route = await nano.route();
    const probe = nano.status();
    status.nano = { route, state: probe.state, reason: probe.reason };
    logger.info(
      'nano',
      `sub-query engine ${route === 'none' ? 'unavailable' : `on ${route}`} (${probe.state}${probe.flavour !== 'none' ? `, ${probe.flavour}` : ''})`,
      probe.reason,
    );

    // Sub-query the goal. Handing Nano the starting page lets step one name a
    // control that exists instead of restating the goal back at us.
    const plan = await nano.decompose(goal, await seedDom(tabId));
    const taskMemory: TaskMemory = {
      goal,
      subObjectives: plan.subObjectives,
      planSource: plan.source,
      currentObjective: plan.subObjectives[0]?.description,
      completedObjectives: [],
      attemptedTargets: [],
      replans: 0,
      step: 0,
    };
    publishPlan(taskMemory);

    logger.info(
      'plan',
      `decomposed goal via ${plan.source} into ${plan.subObjectives.length} sub-objectives`,
      plan.subObjectives.map((s) => s.description).join(' -> '),
    );

    let prevDom: ScrubbedDom | undefined;
    let invalidCount = 0;
    const stagnation = makeStagnationState();

    if (!modelReady) void warmUpLocalModel(settings);

    for (let i = 0; i < settings.maxSteps && !stopRequested; i++) {
      status.step = i + 1;

      // Pre-step termination check
      if (prevDom !== undefined) {
        const signal = checkTermination({
          goal,
          dom: prevDom,
          lastAction: history.at(-1),
          history,
          taskMemory,
        });
        if (signal.done && signal.confidence >= HIGH_CONFIDENCE) {
          logger.info('termination', `early exit before step ${i + 1}: ${signal.reason}`);
          history.push({ action: 'done', summary: signal.reason });
          break;
        }
      }

      taskMemory.step = i + 1;
      const result = await step(tabId, goal, history, settings, taskMemory);
      const action = result.action;
      history.push(action);

      // prevDom for next step
      prevDom = result.postDom;

      // A `done` here means the current *sub-objective* is finished, not the
      // whole goal — so retire it and keep going while any remain.
      if (action.action === 'done') {
        if (advanceObjective(taskMemory, 'model emitted done')) continue;
        break;
      }

      // Post-step completion check only if post-action DOM was observed
      if (
        result.observed &&
        activeIndexOf(taskMemory) === -1
      ) {
        const postSignal = checkTermination({
          goal,
          dom: result.postDom,
          lastAction: action,
          history,
          prevDom: result.preDom,
          taskMemory,
        });

        if (
          postSignal.done &&
          postSignal.confidence >= HIGH_CONFIDENCE
        ) {
          logger.info(
            'termination',
            `post-step exit after step ${i + 1}: ${postSignal.reason}`,
          );

          history.push({
            action: 'done',
            summary: postSignal.reason,
          });

          break;
        }
      }

      // Measure exact DOM change caused by THIS action alone
      const isActionWaitOrDone = action.action === 'wait' || action.action === 'escalate';
      let resultCategory: ActionResultCategory;
      if (action.action === 'invalid') {
        resultCategory = 'failed';
      } else if (!result.observed) {
        resultCategory = 'uncertain';
      } else {
        const stateChanged = fingerprint(result.preDom) !== fingerprint(result.postDom);
        resultCategory = stateChanged ? 'state_changed' : (isActionWaitOrDone ? 'uncertain' : 'no_change');
      }

      taskMemory.lastAction = { action, result: resultCategory };

      if (resultCategory === 'state_changed') {
        taskMemory.attemptedTargets = [];

        if (taskMemory.subObjectives && taskMemory.subObjectives.length > 0) {
          const current = taskMemory.subObjectives[activeIndexOf(taskMemory)];
          if (current) {
            const verdict = await subObjectiveSatisfied(current.description, action, result, history);
            if (verdict.done && !advanceObjective(taskMemory, verdict.why)) {
              // That was the last sub-objective, so the plan is finished by its
              // own definition. Continuing would hand the vision tier an
              // undefined objective and the raw compound goal, which is the
              // thing the decomposition existed to avoid.
              history.push({ action: 'done', summary: `all sub-objectives complete (${verdict.why})` });
              break;
            }
          }
        } else {
          if (action.completedObjective) {
            taskMemory.completedObjectives.push({ description: action.completedObjective, status: 'completed' });
          }
          if (action.currentObjective) {
            taskMemory.currentObjective = action.currentObjective;
          }
        }
      } else if (resultCategory === 'no_change' || resultCategory === 'failed') {
        if ('selector' in action && action.selector) {
          taskMemory.attemptedTargets.push(action.selector);
        }
      }

      // --- Stall handling: re-plan before giving up ---------------------
      //
      // Each of the three conditions in `stallReason` used to end the run
      // outright. They are all the same failure — the active sub-objective
      // cannot be grounded on this page — and that is what Nano can fix, by
      // re-saying the remaining work in terms of the elements that are actually
      // here. The guards still stop the run; they just get one rewrite first.
      if (action.action === 'invalid') invalidCount++;
      else invalidCount = 0;

      const stall = stallReason(action, resultCategory, isActionWaitOrDone, history, invalidCount);
      if (stall) {
        logger.warn('agent', `stalled: ${stall}`);
        if (await tryReplan(taskMemory, result.postDom, history, stall)) {
          // The rewritten plan makes loop detection misleading: the repeated
          // action was aimed at a sub-objective that no longer exists.
          history.length = 0;
          invalidCount = 0;
          continue;
        }
        logger.warn('agent', 'no re-plan available; stopping');
        break;
      }

      // --- Structural stagnation guard ----------------------------------
      if (recordAndCheck(stagnation, prevDom, action, i)) {
        const why = `page stagnant (${stagnation.totalNonWait} non-wait actions, no structural change)`;
        logger.warn('agent', `${why} at step ${i + 1}`);
        if (await tryReplan(taskMemory, result.postDom, history, why)) {
          history.length = 0;
          continue;
        }
        break;
      }

      if (action.action === 'wait') await sleep(Math.min(action.ms, 5_000));
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    status.lastError = errorMsg;
    logger.error('agent', `run aborted: ${errorMsg}`);
  } finally {
    running = false;
    status.running = false;
    ws?.close();
  }
}

/**
 * Why this step counts as a stall, or undefined if it does not.
 *
 * Consolidates the three guards that previously ended a run independently, so
 * the re-plan path has exactly one place to hook into and each guard's threshold
 * stays where it was.
 */
function stallReason(
  action: AgentAction,
  result: ActionResultCategory,
  isWaitOrEscalate: boolean,
  history: readonly AgentAction[],
  invalidCount: number,
): string | undefined {
  if (invalidCount >= 3) return 'three invalid actions in a row';
  if (action.action !== 'invalid' && result === 'no_change' && !isWaitOrEscalate && repeatedTail(history) >= 2) {
    return `${domFingerprint(action)} repeated with no state change`;
  }
  if (repeatedTail(history) >= 3) return `${domFingerprint(action)} chosen three times`;
  return undefined;
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
          // This message originates from a popup click, so it carries the user
          // gesture Chrome wants before it will fetch Gemini Nano's weights.
          // Nowhere inside an agent run can supply that, which is why a
          // 'downloadable' Nano is otherwise reported as simply unavailable.
          void warmUpNano();
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
