/**
 * Adapter over Chrome's built-in Prompt API (Gemini Nano).
 *
 * Everything Chrome-specific and version-specific lives here so the planner
 * above it can be written once and unit-tested against a fake.
 *
 * Three things this file exists to absorb:
 *
 *   1. Two API shapes. Current Chrome exposes a bare `LanguageModel` global
 *      whose `availability()` returns 'available' | 'downloadable' |
 *      'downloading' | 'unavailable'. The origin-trial shape was
 *      `self.ai.languageModel` with 'readily' | 'after-download' | 'no', and a
 *      `systemPrompt` string instead of `initialPrompts`. Both are probed.
 *
 *   2. The paired-sampling rule. In extensions a session must pass *both*
 *      `topK` and `temperature` or neither — passing `temperature` alone throws.
 *      So we read `params()` for a `topK` to go with our temperature 0, and
 *      pass neither if that read fails.
 *
 *   3. Cancellation. A raced timeout that only resolves early leaves the model
 *      generating into a session nobody will destroy, and Nano serialises work
 *      per session — the *next* call then waits on the abandoned one. Every
 *      call here is bounded by an AbortSignal, not just by Promise.race.
 *
 * Nothing in this file touches the network. Gemini Nano is on-device, which is
 * what makes it safe to hand it the scrubbed DOM.
 */

/* ---------------------------------------------------------------- *
 * Chrome Prompt API surface (hand-written: avoids a dependency on
 * @types/dom-chromium-ai, which pins a single API generation)
 * ---------------------------------------------------------------- */

/** Normalised availability, using the modern vocabulary. */
export type NanoState = 'available' | 'downloadable' | 'downloading' | 'unavailable';

/** Which of the two API generations answered the probe. */
export type NanoFlavour = 'language-model' | 'legacy-ai' | 'none';

export interface NanoMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  /** Prefills the start of the reply, to pin output format. */
  prefix?: boolean;
}

interface RawSession {
  prompt: (input: string, opts?: { signal?: AbortSignal; responseConstraint?: object; omitResponseConstraintInput?: boolean }) => Promise<string>;
  clone?: (opts?: { signal?: AbortSignal }) => Promise<RawSession>;
  destroy?: () => void;
}

interface RawCreateOptions {
  initialPrompts?: NanoMessage[];
  /** Legacy `self.ai.languageModel` only. */
  systemPrompt?: string;
  temperature?: number;
  topK?: number;
  signal?: AbortSignal;
  monitor?: (m: EventTarget) => void;
}

interface RawFactory {
  availability?: (opts?: object) => Promise<string>;
  capabilities?: () => Promise<{ available?: string; defaultTopK?: number; maxTopK?: number }>;
  params?: () => Promise<{ defaultTopK?: number; maxTopK?: number; defaultTemperature?: number; maxTemperature?: number } | null>;
  create?: (opts?: RawCreateOptions) => Promise<RawSession>;
}

interface NanoGlobals {
  LanguageModel?: RawFactory;
  ai?: { languageModel?: RawFactory };
}

/* ---------------------------------------------------------------- *
 * Timeouts
 * ---------------------------------------------------------------- */

export const NANO_TIMEOUTS = {
  /** `availability()` is a cheap local lookup; if it hangs, Nano is not usable. */
  probeMs: 2_000,
  /** Session creation compiles nothing but can wait behind a model load. */
  createMs: 15_000,
  /** One planning generation. Bounded because it sits on the run's critical path. */
  promptMs: 9_000,
} as const;

/**
 * Bound `fn` two ways at once, because the Prompt API needs both.
 *
 * `prompt()` and `create()` accept an AbortSignal, so aborting genuinely stops
 * the work — which matters, since Nano serialises per session and an abandoned
 * generation would make the *next* call wait behind it. `availability()` takes
 * no signal at all, so for that one the only bound available is to stop waiting.
 * Racing the timer as well as firing it covers both.
 */
async function bounded<T>(ms: number, fn: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const ctl = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const expiry = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      const err = new DOMException(`nano call exceeded ${ms}ms`, 'TimeoutError');
      ctl.abort(err);
      reject(err);
    }, ms);
  });
  try {
    // Promise.race attaches a handler to both, so a late rejection from the
    // abandoned side cannot surface as an unhandled rejection.
    return await Promise.race([fn(ctl.signal), expiry]);
  } finally {
    clearTimeout(timer);
  }
}

/* ---------------------------------------------------------------- *
 * Probe
 * ---------------------------------------------------------------- */

export interface NanoProbe {
  state: NanoState;
  flavour: NanoFlavour;
  /** Why it is unusable, when it is. Safe to log — no page content. */
  reason?: string;
}

const UNAVAILABLE = (reason: string): NanoProbe => ({ state: 'unavailable', flavour: 'none', reason });

/** Legacy origin-trial vocabulary -> the modern one. */
function normaliseState(raw: string | undefined): NanoState | undefined {
  switch (raw) {
    case 'available':
    case 'readily':
      return 'available';
    case 'downloadable':
    case 'after-download':
      return 'downloadable';
    case 'downloading':
      return 'downloading';
    case 'unavailable':
    case 'no':
      return 'unavailable';
    default:
      return undefined;
  }
}

function factories(scope: object = globalThis): Array<{ flavour: NanoFlavour; api: RawFactory }> {
  const g = scope as NanoGlobals;
  const found: Array<{ flavour: NanoFlavour; api: RawFactory }> = [];
  if (g.LanguageModel && typeof g.LanguageModel.create === 'function') {
    found.push({ flavour: 'language-model', api: g.LanguageModel });
  }
  if (g.ai?.languageModel && typeof g.ai.languageModel.create === 'function') {
    found.push({ flavour: 'legacy-ai', api: g.ai.languageModel });
  }
  return found;
}

/**
 * Is Gemini Nano reachable from *this* JS context?
 *
 * Context matters and is the whole reason `nano-bridge.ts` exists: the Prompt
 * API is exposed to documents, not to workers. A service worker sees no
 * factory at all — or, on some channels, sees the global and then throws when
 * it is used. Both outcomes are reported here as unavailable so the caller can
 * route the work to the offscreen document instead of silently degrading to the
 * rule-based splitter.
 */
export async function probeNano(scope: object = globalThis): Promise<NanoProbe> {
  const candidates = factories(scope);
  if (candidates.length === 0) return UNAVAILABLE('no LanguageModel or ai.languageModel in this context');

  let lastReason = 'availability check returned nothing recognisable';
  for (const { flavour, api } of candidates) {
    try {
      const state = await bounded(NANO_TIMEOUTS.probeMs, async () => {
        if (typeof api.availability === 'function') return normaliseState(await api.availability());
        if (typeof api.capabilities === 'function') return normaliseState((await api.capabilities())?.available);
        // A factory with neither probe is old enough that create() is the probe.
        return 'available' as NanoState;
      });
      if (state && state !== 'unavailable') return { state, flavour };
      if (state === 'unavailable') lastReason = `${flavour} reports the model unavailable on this device`;
    } catch (err) {
      lastReason = `${flavour} availability threw: ${message(err)}`;
    }
  }
  return UNAVAILABLE(lastReason);
}

/* ---------------------------------------------------------------- *
 * Session
 * ---------------------------------------------------------------- */

export interface NanoSessionOptions {
  system: string;
  /** Called with 0..100 while Gemini Nano's weights download. */
  onDownload?: (percent: number) => void;
}

/**
 * A live Nano session, plus the bookkeeping the planner needs.
 *
 * `ask` clones the root session per call. Without that, every planning query in
 * a run would append to one growing context: cost climbs, and a verification
 * answer starts being conditioned on an earlier decomposition. Each query
 * should see the system prompt and nothing else.
 */
export interface NanoSession {
  flavour: NanoFlavour;
  ask(input: string, schema?: object): Promise<string>;
  destroy(): void;
}

/**
 * Open a session, downloading the model first if Chrome says it needs to.
 *
 * `create()` is what triggers that download, and Chrome wants a user gesture
 * for it — so the caller should reach this from the popup's warm-up, not from
 * the middle of an agent run. From an agent run a 'downloadable' state is
 * reported as a failure rather than kicking off a multi-minute download the
 * user did not ask for.
 */
export async function openNanoSession(
  opts: NanoSessionOptions,
  scope: object = globalThis,
  { allowDownload = false }: { allowDownload?: boolean } = {},
): Promise<NanoSession | null> {
  const probe = await probeNano(scope);
  if (probe.state === 'unavailable') return null;
  if (probe.state !== 'available' && !allowDownload) return null;

  const entry = factories(scope).find((f) => f.flavour === probe.flavour);
  if (!entry?.api.create) return null;
  const { api, flavour } = entry;

  // Extensions must pass both sampling knobs or neither. Temperature 0 is what
  // we want for planning; the topK to pair it with has to come from params().
  const sampling = await samplingFor(api);

  const root = await bounded(NANO_TIMEOUTS.createMs, (signal) =>
    api.create!({
      ...(flavour === 'legacy-ai'
        ? { systemPrompt: opts.system }
        : { initialPrompts: [{ role: 'system', content: opts.system }] }),
      ...sampling,
      signal,
      ...(opts.onDownload
        ? {
            monitor: (m: EventTarget) =>
              m.addEventListener('downloadprogress', (ev: Event) => {
                const loaded = (ev as Event & { loaded?: number }).loaded;
                if (typeof loaded === 'number') opts.onDownload!(Math.round(loaded * 100));
              }),
          }
        : {}),
    }),
  );

  let destroyed = false;

  return {
    flavour,
    async ask(input: string, schema?: object): Promise<string> {
      if (destroyed) throw new Error('nano session already destroyed');
      return bounded(NANO_TIMEOUTS.promptMs, async (signal) => {
        // Clone so each query starts from the system prompt alone. A factory
        // without clone() still works; its context just accumulates.
        const turn = root.clone ? await root.clone({ signal }).catch(() => root) : root;
        try {
          // `responseConstraint` is the reliable path — Chrome constrains
          // decoding to the schema, so there is no prose to scrape off. Older
          // builds ignore or reject the option, hence the retry.
          if (schema) {
            try {
              return await turn.prompt(input, { signal, responseConstraint: schema });
            } catch (err) {
              if (isAbort(err)) throw err;
            }
          }
          return await turn.prompt(input, { signal });
        } finally {
          if (turn !== root) turn.destroy?.();
        }
      });
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      root.destroy?.();
    },
  };
}

async function samplingFor(api: RawFactory): Promise<{ temperature: number; topK: number } | Record<string, never>> {
  try {
    const p = (await api.params?.()) ?? (await api.capabilities?.());
    const topK = p && 'defaultTopK' in p ? p.defaultTopK : undefined;
    if (typeof topK === 'number' && topK > 0) return { temperature: 0, topK };
  } catch {
    // Fall through: neither knob is safer than one.
  }
  return {};
}

const isAbort = (err: unknown): boolean =>
  err instanceof Error && (err.name === 'AbortError' || err.name === 'TimeoutError');

export const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));
