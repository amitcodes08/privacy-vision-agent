/**
 * Gemini Nano sub-query engine.
 *
 * This is the layer the architecture now leans on: a complex user goal is not
 * executed directly, it is *sub-queried* — broken into atomic, page-grounded
 * objectives that the vision tier can actually act on one at a time.
 *
 * Three jobs, all on-device, all bounded, all with a deterministic fallback:
 *
 *   decompose   goal            -> ordered atomic sub-objectives   (once, at run start)
 *   replan      goal + page     -> rewritten remaining objectives  (when a step stalls)
 *   verify      objective + page -> did that sub-objective land?    (per sub-objective)
 *
 * `replan` is the piece that makes multi-step tasks finish. A decomposition made
 * before the agent has seen a single page is a guess: "Filter by price under
 * $500" is not an action any site offers verbatim. When a sub-objective cannot
 * be grounded, Nano gets to look at what the page actually offers and re-say the
 * remaining work in those terms.
 *
 * Privacy: Gemini Nano runs on-device, so handing it a page digest costs no
 * network egress. The digest is nevertheless built from the *already scrubbed*
 * DOM and drops field values entirely — see `digestPage`. Raw pixels never come
 * near this file.
 */
import type { AgentAction, ScrubbedDom, TaskObjective } from '@shared/types';
import { message, openNanoSession, probeNano, type NanoProbe, type NanoSession } from '~/ai/nano-session';

export type PlanSource = 'gemini-nano' | 'gemini-nano-replan' | 'local-rules';

export interface DecomposedPlan {
  subObjectives: TaskObjective[];
  source: PlanSource;
}

/** Hard ceiling on plan length — a 12-step plan cannot finish in `maxSteps`. */
const MAX_SUB_OBJECTIVES = 6;
const MAX_OBJECTIVE_CHARS = 120;

/* ---------------------------------------------------------------- *
 * Session lifecycle
 * ---------------------------------------------------------------- */

let cached: NanoSession | null = null;
let cachedProbe: NanoProbe | null = null;
/** Single-flight, so three planning calls in one step do not open three sessions. */
let opening: Promise<NanoSession | null> | null = null;

const SYSTEM_PROMPT = [
  'You are the planning stage of a web browser automation agent.',
  'You never interact with the page yourself. You only restate work as short, atomic steps',
  'that a separate vision agent will carry out one at a time.',
  'An atomic step names ONE interaction with ONE element: "Click the search box",',
  '"Type \'wireless mouse\'", "Click Add to Cart", "Click the first result".',
  'Never invent steps for work the user did not ask for. Never mention logging in unless asked.',
  'Answer with JSON only. No prose, no markdown, no backticks.',
].join('\n');

/**
 * Open (or reuse) the planning session for this context.
 *
 * Returns null whenever Nano is unreachable — including from a service worker,
 * where the Prompt API is not exposed at all. Callers must treat null as
 * "use the deterministic fallback", never as an error.
 */
export async function nanoPlanner(opts: { allowDownload?: boolean } = {}): Promise<NanoSession | null> {
  if (cached) return cached;
  opening ??= (async () => {
    try {
      cachedProbe = await probeNano();
      cached = await openNanoSession({ system: SYSTEM_PROMPT }, globalThis, {
        allowDownload: opts.allowDownload ?? false,
      });
      return cached;
    } catch (err) {
      cachedProbe = { state: 'unavailable', flavour: 'none', reason: message(err) };
      return null;
    } finally {
      opening = null;
    }
  })();
  return opening;
}

/** What the last probe saw. Surfaced in the popup so "why no Nano" is answerable. */
export function nanoStatus(): NanoProbe {
  return cachedProbe ?? { state: 'unavailable', flavour: 'none', reason: 'not probed yet' };
}

/** Drop the session. Called at the end of a run and by tests. */
export function closeNanoPlanner(): void {
  cached?.destroy();
  cached = null;
  cachedProbe = null;
  opening = null;
}

/**
 * Force a session in. Used by the offscreen host to share one session across
 * message kinds, and by unit tests to inject a fake.
 */
export function setNanoPlanner(session: NanoSession | null, probe?: NanoProbe): void {
  cached = session;
  if (probe) cachedProbe = probe;
  else if (session) cachedProbe = { state: 'available', flavour: session.flavour };
}

/** Kept for the popup/warm-up path: does this context have a usable Nano? */
export async function isGeminiNanoAvailable(): Promise<boolean> {
  return (await probeNano()).state === 'available';
}

/* ---------------------------------------------------------------- *
 * 1. Decompose
 * ---------------------------------------------------------------- */

const DECOMPOSE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['steps'],
  properties: {
    steps: {
      type: 'array',
      minItems: 1,
      maxItems: MAX_SUB_OBJECTIVES,
      items: { type: 'string' },
    },
  },
} as const;

/**
 * Break a user goal into ordered sub-objectives.
 *
 * `dom` is optional and only present when the agent already has a page in hand;
 * with it, Nano can phrase step one in terms of something that exists.
 */
export async function decomposeGoal(goal: string, dom?: ScrubbedDom): Promise<DecomposedPlan> {
  const trimmed = goal.trim();
  if (!trimmed) return { subObjectives: [], source: 'local-rules' };

  const session = await nanoPlanner();
  if (session) {
    const steps = await askForSteps(session, [
      `USER GOAL: ${trimmed}`,
      dom ? `\nCURRENT PAGE:\n${digestPage(dom)}` : '',
      '',
      `Break the goal into 1-${MAX_SUB_OBJECTIVES} atomic steps, in the order they must happen.`,
      'If the goal is already a single interaction, return exactly one step.',
      'Respond as {"steps":["...","..."]}.',
    ].join('\n'));

    if (steps.length > 0) return { subObjectives: toObjectives(steps), source: 'gemini-nano' };
  }

  return { subObjectives: decomposeWithRules(trimmed), source: 'local-rules' };
}

/* ---------------------------------------------------------------- *
 * 2. Replan
 * ---------------------------------------------------------------- */

export interface ReplanInput {
  /** The original user goal — the thing that must still be true at the end. */
  goal: string;
  /** Sub-objectives not yet completed, in order. The first is the stuck one. */
  remaining: readonly TaskObjective[];
  /** Sub-objectives already satisfied, for context. */
  completed?: readonly TaskObjective[];
  dom: ScrubbedDom;
  history?: readonly AgentAction[];
  /** Why we are replanning, e.g. "no actionable element for 2 steps". */
  reason: string;
}

export interface ReplanResult {
  subObjectives: TaskObjective[];
  source: PlanSource;
  /** True when Nano actually produced a different plan. */
  changed: boolean;
}

/**
 * Re-say the remaining work in terms of what this page actually offers.
 *
 * Returns `changed: false` (and the input untouched) whenever Nano is
 * unavailable or its answer is not materially different — the caller should then
 * fall through to whatever it would have done anyway, usually stopping. There is
 * deliberately no rule-based fallback here: a deterministic rewrite of a stuck
 * plan would just be the same plan.
 */
export async function replanFromPage(input: ReplanInput): Promise<ReplanResult> {
  const unchanged: ReplanResult = { subObjectives: [...input.remaining], source: 'local-rules', changed: false };

  const session = await nanoPlanner();
  if (!session) return unchanged;

  const past = (input.history ?? [])
    .slice(-4)
    .map((h) => ('selector' in h && h.selector ? `${h.action}(${h.selector})` : h.action))
    .join(', ');

  const steps = await askForSteps(session, [
    `USER GOAL: ${input.goal}`,
    input.completed?.length ? `ALREADY DONE: ${input.completed.map((o) => o.description).join(' -> ')}` : '',
    `STUCK ON: ${input.remaining[0]?.description ?? input.goal}`,
    `WHY: ${input.reason}`,
    past ? `RECENT ACTIONS THAT DID NOT HELP: ${past}` : '',
    '',
    'CURRENT PAGE:',
    digestPage(input.dom),
    '',
    'Rewrite ONLY the work that still remains, as atomic steps that use the elements',
    'listed above. Reuse an element\'s exact label text so the vision agent can find it.',
    'Do not repeat steps under ALREADY DONE. Do not repeat the actions that did not help.',
    'If nothing on this page can advance the goal, return one step describing where to go instead.',
    'Respond as {"steps":["...","..."]}.',
  ].join('\n'));

  if (steps.length === 0) return unchanged;

  const before = input.remaining.map((o) => o.description.toLowerCase()).join('|');
  const after = steps.map((s) => s.toLowerCase()).join('|');
  if (before === after) return unchanged;

  return { subObjectives: toObjectives(steps), source: 'gemini-nano-replan', changed: true };
}

/* ---------------------------------------------------------------- *
 * 3. Verify
 * ---------------------------------------------------------------- */

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['satisfied', 'reason'],
  properties: {
    satisfied: { type: 'boolean' },
    reason: { type: 'string' },
  },
} as const;

export interface VerifyInput {
  objective: string;
  dom: ScrubbedDom;
  lastAction?: AgentAction;
}

export interface VerifyResult {
  satisfied: boolean;
  /** 0 when Nano did not answer, so the caller keeps its own evidence. */
  confidence: number;
  reason: string;
  source: 'gemini-nano' | 'unavailable';
}

const NO_VERDICT: VerifyResult = {
  satisfied: false,
  confidence: 0,
  reason: 'nano unavailable',
  source: 'unavailable',
};

/**
 * Ask Nano whether one sub-objective is now satisfied by the page.
 *
 * Confidence is capped below the caller's HIGH_CONFIDENCE gate on purpose: a
 * 2B on-device model saying "yes" is evidence, not proof, and advancing a
 * sub-objective early strands the rest of the plan. The caller combines this
 * with the deterministic DOM check.
 */
export async function verifySubObjective(input: VerifyInput): Promise<VerifyResult> {
  const session = await nanoPlanner();
  if (!session) return NO_VERDICT;

  const raw = await session
    .ask(
      [
        `STEP TO CHECK: ${input.objective}`,
        input.lastAction ? `JUST EXECUTED: ${describeAction(input.lastAction)}` : '',
        '',
        'PAGE AFTER THAT ACTION:',
        digestPage(input.dom),
        '',
        'Is the STEP TO CHECK now finished, judging only by the page above?',
        'Answer false if you are unsure or if the page merely shows the step in progress.',
        'Respond as {"satisfied":true|false,"reason":"one short sentence"}.',
      ].join('\n'),
      VERIFY_SCHEMA,
    )
    .catch((err) => {
      warn(`verify failed: ${message(err)}`);
      return '';
    });

  const parsed = parseObject(raw);
  if (!parsed || typeof parsed.satisfied !== 'boolean') return NO_VERDICT;

  return {
    satisfied: parsed.satisfied,
    confidence: parsed.satisfied ? 0.7 : 0.55,
    reason: typeof parsed.reason === 'string' ? parsed.reason.slice(0, 160) : 'nano verdict',
    source: 'gemini-nano',
  };
}

/* ---------------------------------------------------------------- *
 * Page digest
 * ---------------------------------------------------------------- */

/** How many elements Nano is shown. It is a small model; a long list hurts. */
const DIGEST_ELEMENTS = 24;

/**
 * A compact, text-only description of the page for the planner.
 *
 * Built from the already-scrubbed DOM, and deliberately narrower than that:
 *
 *   - `value` is never included. A scrubbed node keeps its own typed value
 *     (`dom-scrubber` only masks nodes it flagged), so echoing values here would
 *     widen what the planner sees for no planning benefit.
 *   - `href` is reduced to nothing; only the page origin appears, once.
 *   - any node carrying a `redacted` reason is described by tag and role only.
 */
export function digestPage(dom: ScrubbedDom): string {
  const lines: string[] = [`url: ${dom.origin}`, `title: ${trunc(dom.title, 80)}`];

  const interactive = dom.nodes.filter((n) => n.visible && !n.disabled).slice(0, DIGEST_ELEMENTS);
  for (const n of interactive) {
    if (n.redacted?.length) {
      lines.push(`- ${n.tag}${n.role ? `[${n.role}]` : ''} (sensitive field, contents withheld)`);
      continue;
    }
    const label = n.label ?? n.text ?? n.placeholder ?? n.name;
    lines.push(
      `- ${n.tag}${n.type ? `[${n.type}]` : ''}${label ? ` "${trunc(label, 60)}"` : ''}`,
    );
  }
  if (dom.nodes.length > interactive.length) {
    lines.push(`(+${dom.nodes.length - interactive.length} more elements not listed)`);
  }
  return lines.join('\n');
}

const describeAction = (a: AgentAction): string =>
  'selector' in a && a.selector ? `${a.action} on ${a.selector}` : a.action;

/* ---------------------------------------------------------------- *
 * Deterministic fallback
 * ---------------------------------------------------------------- */

/**
 * Deterministic clause-splitting decomposer.
 *
 * Runs when Nano is absent (no Chrome built-in AI, older Chrome, unsupported
 * hardware, or the model still downloading). Zero network, zero download,
 * instant — it just cannot rewrite a goal, only split one.
 */
export function decomposeWithRules(goal: string): TaskObjective[] {
  const SPLIT_REGEX = /(?:\s*(?:and\s+then|then|after\s+that|afterwards|and\s+also|followed\s+by|->)\s*|\s*;\s*)/i;

  let clauses = goal.split(SPLIT_REGEX).map((c) => c.trim()).filter(Boolean);

  // If not split by explicit 'then' / 'and then', check for compound 'and' with action verbs
  if (clauses.length === 1) {
    const compoundAndSplit = splitCompoundAnd(goal);
    if (compoundAndSplit.length > 1) {
      clauses = compoundAndSplit;
    }
  }

  const subObjectives = toObjectives(clauses.map(normalizeClause).filter(Boolean));

  return subObjectives.length > 0 ? subObjectives : [{ id: 1, description: goal, status: 'active' }];
}

/**
 * Splits compound sentences joined by "and" when followed by an action verb
 * e.g., "Search for shoes and add to cart" -> ["Search for shoes", "Add to cart"]
 */
function splitCompoundAnd(text: string): string[] {
  const ACTION_VERBS = '(?:click|press|open|fill|type|enter|input|select|choose|add|navigate|go\\s+to|visit|check\\s*out|submit|proceed|find|search|scroll)';
  const pattern = new RegExp(`\\s+(?:,\\s*)?and\\s+(?=${ACTION_VERBS}\\b)`, 'i');

  const parts = text.split(pattern).map((s) => s.trim()).filter(Boolean);
  return parts.length > 1 ? parts : [text];
}

function normalizeClause(clause: string): string {
  const trimmed = clause.replace(/^[;,.\s]+|[;,.\s]+$/g, '');
  if (!trimmed) return '';
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
}

/* ---------------------------------------------------------------- *
 * Shared plumbing
 * ---------------------------------------------------------------- */

/**
 * One `{"steps":[...]}` query, with every failure folded into an empty array.
 *
 * A planner that throws would abort the agent run over an optional
 * optimisation, so nothing here propagates.
 */
async function askForSteps(session: NanoSession, prompt: string): Promise<string[]> {
  const raw = await session.ask(prompt, DECOMPOSE_SCHEMA).catch((err) => {
    warn(`plan query failed: ${message(err)}`);
    return '';
  });

  const parsed = parseObject(raw);
  const steps = parsed && Array.isArray(parsed.steps) ? parsed.steps : parseBareArray(raw);
  if (!steps) return [];

  return sanitiseSteps(steps);
}

/** Trim, drop junk, dedupe, cap. A model that repeats itself must not create a loop. */
export function sanitiseSteps(raw: readonly unknown[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of raw) {
    if (typeof item !== 'string') continue;
    const step = item.replace(/^\s*(?:step\s*)?\d+[.)]?\s*/i, '').trim().slice(0, MAX_OBJECTIVE_CHARS);
    if (step.length < 2) continue;
    const key = step.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(step.charAt(0).toUpperCase() + step.slice(1));
    if (out.length >= MAX_SUB_OBJECTIVES) break;
  }
  return out;
}

/** Descriptions -> objectives, with the first one active. */
function toObjectives(steps: readonly string[]): TaskObjective[] {
  return steps.map((description, idx) => ({
    id: idx + 1,
    description,
    status: idx === 0 ? 'active' : 'pending',
  }));
}

/** First balanced JSON object in the text, tolerant of fences and stray prose. */
function parseObject(raw: string): Record<string, unknown> | null {
  if (!raw) return null;
  const cleaned = raw.replace(/```(?:json)?/gi, '');
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start === -1 || end <= start) return null;
  try {
    const json = JSON.parse(cleaned.slice(start, end + 1)) as unknown;
    return typeof json === 'object' && json !== null ? (json as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/**
 * A bare `["a","b"]` reply, which is what a build that ignored
 * `responseConstraint` tends to produce against the older prompt shape.
 */
function parseBareArray(raw: string): unknown[] | null {
  if (!raw) return null;
  const cleaned = raw.replace(/```(?:json)?/gi, '');
  const start = cleaned.indexOf('[');
  const end = cleaned.lastIndexOf(']');
  if (start === -1 || end <= start) return null;
  try {
    const json = JSON.parse(cleaned.slice(start, end + 1)) as unknown;
    return Array.isArray(json) ? json : null;
  } catch {
    return null;
  }
}

const trunc = (s: string, n: number): string => (s.length > n ? `${s.slice(0, n)}…` : s);

const warn = (msg: string): void => {
  console.debug(`[nano-query-planner] ${msg}`);
};
