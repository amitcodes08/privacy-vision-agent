/**
 * Turning local model text into a trustworthy action.
 *
 * Deliberately free of `@huggingface/transformers` imports so it can be unit
 * tested (and read by the service worker) without pulling in onnxruntime.
 *
 * The hard part is not parsing JSON — it is deciding how much to trust what
 * came back. A small VLM gets the *intent* right far more often than it gets
 * an exact CSS selector right, so most of this file is about resolving what
 * the model meant back to a real element before scoring it. Every element we
 * recover here is one escalation that does not happen.
 */
import { type AgentAction, type AgentDecision, type ScrubbedDom, type ScrubbedNode, type ValueToken } from '@shared/types';
import { inferValueType, rankCandidates, rankOf, type Ranking } from './local-planner';

/** How many elements the prompt may list. Small models degrade with long lists. */
const PROMPT_BUDGET = 36;
/** Ranked elements guaranteed a slot, however long the page is. */
const GUARANTEED_RELEVANT = 18;

/**
 * Compact, token-cheap page description for the local model.
 *
 * Element choice is the whole game here. `dom.nodes` holds up to
 * `maxDomNodes` (120) entries, so slicing the first N in DOM order could omit
 * the one element the goal is about entirely — the model then had no way to
 * answer correctly and looked "unsure" for a reason that was our fault. The
 * ranker guarantees goal-relevant elements a slot; the rest of the budget is
 * page context.
 *
 * Emission stays in DOM order so the list lines up with the screenshot the
 * model is looking at. Relevance is a marker, not a reordering — the model
 * still decides.
 */
export function buildPrompt(
  goal: string,
  dom: ScrubbedDom,
  history: AgentAction[] = [],
  ranking?: Ranking,
  taskMemory?: import('@shared/types').TaskMemory,
): string {
  const ranked =
    ranking ??
    rankCandidates({
      goal,
      dom,
      history,
      taskMemory,
    });

  const activeObjective =
    taskMemory?.currentObjective?.trim() || goal;

  const relevant = new Set(
    ranked.candidates
      .slice(0, GUARANTEED_RELEVANT)
      .map((c) => c.node.id),
  );

  const chosen = new Map<number, ScrubbedNode>();

  for (const c of ranked.candidates.slice(0, GUARANTEED_RELEVANT)) {
    chosen.set(c.node.id, c.node);
  }

  for (const n of dom.nodes) {
    if (chosen.size >= PROMPT_BUDGET) break;
    if (n.visible && !n.disabled) {
      chosen.set(n.id, n);
    }
  }

  const lines = [...chosen.values()]
    .sort((a, b) => a.id - b.id)
    .map((n) => {
      const bits = [
        n.tag,
        n.type && `type=${n.type}`,
        n.role && `role=${n.role}`,
        n.label && `label="${trunc(n.label)}"`,
        n.text && `text="${trunc(n.text)}"`,
        n.placeholder && `placeholder="${trunc(n.placeholder)}"`,
        n.context && `context="${trunc(n.context)}"`,
        n.value && `value="${trunc(n.value)}"`,
      ]
        .filter(Boolean)
        .join(' ');

      return `${n.id}: ${bits}${
        relevant.has(n.id) ? '  <-- mentions your goal' : ''
      }`;
    });

  const past = history
    .slice(-4)
    .map((h) =>
      'selector' in h
        ? `${h.action}(${h.selector})`
        : h.action,
    )
    .join(', ');

  const modeGuidance =
    ranked.mode === 'search'
      ? [
          'MODE: SEARCH',
          'Choose the search field that can accept the requested query.',
          ranked.searchQuery
            ? `SEARCH QUERY: "${ranked.searchQuery}"`
            : '',
          'Do not choose unrelated links, navigation, ads, or page text.',
        ]
      : ranked.mode === 'target'
        ? [
            'MODE: TARGET SELECTION',
            'Choose the visible actionable element that best identifies the requested target.',
            'Prefer the requested object itself over accessories, replacements, covers, cables, ads, or unrelated navigation.',
            'Use both the screenshot and the element text.',
          ]
        : [
            'MODE: UI ACTION',
            'Choose the single actionable element that best matches the active objective.',
          ];

  return [
    'You are a browser action selector.',
    'Use the screenshot as the primary visual source and the element list as grounding.',
    '',
    `OVERALL GOAL: ${goal}`,
    `ACTIVE OBJECTIVE: ${activeObjective}`,
    '',
    ...modeGuidance,
    '',
    'IMPORTANT:',
    '- Solve ONLY the ACTIVE OBJECTIVE.',
    '- Perform exactly ONE action.',
    '- Never invent an element id.',
    '- Never choose an element merely because one word matches.',
    '- Prefer a semantically correct target over a generic page control.',
    '- Do not repeat an action already attempted unsuccessfully.',
    '- Do NOT return "done". Completion is verified by the controller.',
    '- Do NOT put explanations outside JSON.',
    '',
    'ELEMENTS:',
    lines.join('\n'),
    '',
    'RECENT ACTIONS:',
    past ? `${past} — do not repeat these.` : 'none',
    '',
    taskMemory?.attemptedTargets?.length
      ? `FAILED TARGETS: ${taskMemory.attemptedTargets.join(', ')}`
      : '',
    '',
    'OUTPUT EXACTLY ONE JSON OBJECT:',
    '{"action":"click","id":123}',
    '{"action":"fill","id":123,"valueType":"LITERAL","value":"text"}',
    '{"action":"scroll","deltaY":600}',
    '',
    'Allowed actions: click, fill, scroll, escalate',
    'For click: "id" is required.',
    'For fill: "id", "valueType", and "value" are required.',
    'For scroll: "deltaY" is optional.',
    'Return JSON only.',
  ]
    .filter(Boolean)
    .join('\n');
}

const trunc = (s: string, n = 48) => (s.length > n ? `${s.slice(0, n)}…` : s);

const ACTION_SYNONYMS: Record<string, AgentAction['action']> = {
  click: 'click', press: 'click', tap: 'click', push: 'click', select: 'click', open: 'click',
  fill: 'fill', type: 'fill', enter: 'fill', input: 'fill', write: 'fill', set: 'fill',
  scroll: 'scroll', navigate: 'navigate', goto: 'navigate', go: 'navigate',
  wait: 'wait', done: 'done', finish: 'done', complete: 'done', stop: 'done', end: 'done',
  escalate: 'escalate', unsure: 'escalate', unknown: 'escalate',
};

const VALUE_TOKENS = new Set<ValueToken>([
  'USER_EMAIL', 'USER_FULL_NAME', 'USER_PHONE', 'USER_ADDRESS', 'USER_PASSWORD', 'OTP_CODE', 'LITERAL',
]);

export interface ParseContext {
  /** Enables the corroboration bonus below. */
  goal?: string;
  history?: AgentAction[];
  /** Reuse the ranking already computed for the prompt. */
  ranking?: Ranking;
  taskMemory?: import('@shared/types').TaskMemory;
}

/**
 * Parse the model's text into an action and score how much we trust it.
 * The score is structural, not probabilistic: an element that does not exist
 * on the page is worthless no matter how confident the logits were.
 */
export function parseAction(
  raw: string,
  dom: ScrubbedDom,
  ctx: ParseContext = {},
): { action: AgentAction; confidence: number } {
  const json = extractJson(raw);
  if (!json || typeof json !== 'object') {
    // Before giving up, check whether the raw text is a strong explicit completion statement.
    if (parseDoneFromText(raw)) {
      // Confidence 0.65: above the confidence threshold so the decision is
      // treated as usable (not vlmUnusable), but below HIGH_CONFIDENCE so it
      // still requires DOM corroboration before the loop stops unconditionally.
      return { action: { action: 'done', summary: raw.trim().slice(0, 200) }, confidence: 0.65 };
    }
    // DO NOT emit `escalate` as a generic "I don't know".
    // Unparseable output should result in an invalid action (confidence 0) so the planner can take over.
    return { action: { action: 'invalid', reason: `no JSON in local output (output: "${raw.trim().slice(0, 100)}")` }, confidence: 0 };
  }

  const { action, match } = normalize(json as Record<string, unknown>, dom);
  if (!action) {
    const attempted = (json as any).action || 'unknown';
    return { action: { action: 'invalid', reason: `invalid action emitted: "${attempted}"` }, confidence: 0 };
  }
  if (action.action === 'escalate') return { action, confidence: 0 };
  if (action.action === 'done') return { action, confidence: 0.7 };
  if (action.action === 'wait') return { action, confidence: 0.6 };
  if (action.action === 'navigate') return { action, confidence: 0.6 };
  if (action.action === 'scroll' && !action.selector) return { action, confidence: 0.65 };
  if (action.action === 'invalid') return { action, confidence: 0 };


  let confidence = 0.55;
  if (match === 'exact') confidence += 0.32;
  else if (match === 'repaired') confidence += 0.2;

  const node = findNode(dom, action);
  if (node?.disabled) confidence -= 0.25;
  if (action.action === 'fill' && !action.valueType) confidence -= 0.2;

  // Corroboration: the keyword ranker looked at the same page independently.
  // When it would have picked the same element, that is real evidence the
  // model read the page rather than guessing — and it is what keeps a correct
  // but hesitant local decision from becoming a network round trip. The model
  // still chose; this only vouches for the choice.
  if (ctx.goal && node) {
    const ranking = ctx.ranking ?? rankCandidates({ goal: ctx.goal, dom, history: ctx.history ?? [] });
    const at = rankOf(ranking, node.selector);
    if (at === 0) confidence += 0.12;
    else if (at !== undefined && at < 5) confidence += 0.06;
  }

  return { action, confidence: Math.max(0, Math.min(1, confidence)) };
}

/**
 * Detect natural-language completion statements that small models emit
 * *instead of* a JSON `done` object.
 *
 * Deliberately narrow — we only match strong, unambiguous explicit completion
 * phrases. Arbitrary assistant text must NOT become `done`. A false negative
 * (missed done) is far cheaper than a false positive (premature stop).
 *
 * Exported for unit tests; not part of the public decision-making API.
 */
export function parseDoneFromText(raw: string): boolean {
  const t = raw.trim();
  if (!t || t.length < 10) return false;

  // The patterns below are anchored to the start of the response (optionally
  // preceded by whitespace/quotes) so they do not fire on a sentence that
  // happens to contain "goal is satisfied" as a subordinate clause.
  const DONE_PATTERNS = [
    /^(?:the )?goal is already satisfied/i,
    /^the goal is satisfied/i,
    /^the goal is already complete/i,
    /^(?:the )?task is (?:already )?(?:complete|done|finished)/i,
    /^the requested (?:task|action|goal) (?:has been|is) (?:complete|done|finished|accomplished)/i,
    // The specific phrase observed in the bug report:
    /^the goal is already satisfied by (?:the )?(?:current|what you see)/i,
    // Prompt echo: the model restated step-1 output verbatim
    /^(?:the )?goal is already fully satisfied/i,
  ];

  const lower = t.toLowerCase();
  return DONE_PATTERNS.some((re) => re.test(lower));
}

/**
 * Strip semantically-inconsistent fields from a cloud-sourced AgentDecision.
 *
 * The cloud occasionally returns a `click` action that includes `valueType`
 * and `value` fields intended for `fill`. These fields are ignored by the
 * browser executor but signal a confused model response. Sanitising here
 * prevents accidental misuse if the executor is ever extended.
 *
 * Safe to call on local decisions too — it is a no-op for well-formed actions.
 */
export function sanitiseCloudAction(decision: AgentDecision): AgentDecision {
  const a = decision.action;
  if (a.action !== 'click') return decision;

  // A click action must not carry fill-specific fields.
  const raw = a as unknown as Record<string, unknown>;
  const hasFillFields = 'valueType' in raw || ('value' in raw && typeof raw.value === 'string');
  if (!hasFillFields) return decision;

  // Reject malformed combinations instead of silently dropping fields.
  return {
    ...decision,
    action: {
      action: 'invalid',
      reason: 'malformed cloud response: click action contained fill fields',
    },
  };
}

export type Match = 'exact' | 'repaired' | 'unresolved' | 'none';

/** Coerce whatever the model produced into a valid `AgentAction`. */
function normalize(obj: Record<string, unknown>, dom: ScrubbedDom): { action: AgentAction | null; match: Match } {
  const kindRaw = String(obj.action ?? obj.type ?? obj.name ?? '').toLowerCase().trim();
  const kind = ACTION_SYNONYMS[kindRaw];
  if (!kind) return { action: null, match: 'none' };

  const memFields = {
    ...(obj.completedObjective ? { completedObjective: str(obj.completedObjective) } : {}),
    ...(obj.currentObjective ? { currentObjective: str(obj.currentObjective) } : {}),
  };

  if (kind === 'done') {
    const summary = str(obj.summary ?? obj.reason);
    return { action: { action: 'done', ...(summary ? { summary } : {}), ...memFields }, match: 'none' };
  }
  if (kind === 'escalate') {
    return { action: { action: 'escalate', reason: str(obj.reason), ...memFields }, match: 'none' };
  }
  if (kind === 'wait') {
    const ms = num(obj.ms ?? obj.duration) ?? 500;
    return { action: { action: 'wait', ms: Math.max(0, Math.min(5_000, ms)), ...memFields }, match: 'none' };
  }
  if (kind === 'navigate') {
    const url = str(obj.url ?? obj.href);
    if (!url || !/^https?:\/\//i.test(url)) return { action: null, match: 'none' };
    return { action: { action: 'navigate', url, reason: str(obj.reason), ...memFields }, match: 'none' };
  }

  const resolved = resolveTarget(obj, dom);

  if (kind === 'scroll') {
    const deltaY = num(obj.deltaY ?? obj.delta ?? obj.amount);
    if (resolved.node) {
      return { action: { action: 'scroll', selector: resolved.node.selector, reason: str(obj.reason), ...memFields }, match: resolved.match };
    }
    return {
      action: { action: 'scroll', deltaY: deltaY ?? Math.round(dom.viewport.height * 0.8), reason: str(obj.reason), ...memFields },
      match: 'none',
    };
  }

  // click / fill both need a target element.
  if (!resolved.node) {
    const selector = str(obj.selector ?? obj.css ?? obj.element);
    if (!selector) return { action: null, match: 'none' };
    return {
      action: {
        action: 'invalid',
        reason: `target element "${selector}" not found in DOM`,
        ...memFields
      },
      match: 'none',
    };
  }

  const node = resolved.node;
  if (kind === 'fill') {
    const isFillable = node.tag === 'input' || node.tag === 'textarea' || 
                       node.role === 'textbox' || node.role === 'combobox' || node.role === 'searchbox';
    
    if (!isFillable) {
      return { action: { action: 'invalid', reason: 'fill action targeted a non-input element' }, match: 'none' };
    }

    const valueType = valueTokenOf(obj) ?? inferValueType(node);
    return {
      action: {
        action: 'fill',
        selector: node.selector,
        valueType,
        reason: str(obj.reason),
        ...(valueType === 'LITERAL' ? literal(obj) : {}),
        ...(bool(obj.submit) ? { submit: true } : {}),
        ...memFields,
      },
      match: resolved.match,
    };
  }
  
  if ('valueType' in obj || 'valuetype' in obj || 'value_type' in obj || ('value' in obj && typeof obj.value === 'string')) {
    return { action: { action: 'invalid', reason: 'click action contained fill fields' }, match: 'none' };
  }

  return { action: { action: 'click', selector: node.selector, reason: str(obj.reason), ...memFields }, match: resolved.match };
}

/**
 * Find the element the model meant. In order of reliability: the numeric id we
 * printed in the prompt, an exact selector, then a label/text lookup.
 */
export function resolveTarget(obj: Record<string, unknown>, dom: ScrubbedDom): { node?: ScrubbedNode; match: Match } {
  const id = num(obj.id ?? obj.element_id ?? obj.elementId ?? obj.index);
  if (id !== undefined) {
    const byId = dom.nodes.find((n) => n.id === id);
    if (byId) return { node: byId, match: 'exact' };
  }

  const selector = str(obj.selector ?? obj.css ?? obj.element ?? obj.target);
  if (selector) {
    const exact = dom.nodes.find((n) => n.selector === selector);
    if (exact) return { node: exact, match: 'exact' };

    // A bare integer in the selector slot is an id in disguise.
    if (/^\d+$/.test(selector)) {
      const byNumericSelector = dom.nodes.find((n) => n.id === Number(selector));
      if (byNumericSelector) return { node: byNumericSelector, match: 'exact' };
    }

    const needle = selector.replace(/^[#.]/, '').toLowerCase().trim();
    if (needle) {
      const byIdent = dom.nodes.find(
        (n) => n.name?.toLowerCase() === needle || n.selector.toLowerCase().endsWith(`#${needle}`),
      );
      if (byIdent) return { node: byIdent, match: 'repaired' };
    }
  }

  // Label/text the model may have used instead of a selector.
  const label = str(obj.label ?? obj.text ?? obj.value_label) ?? selector;
  if (label) {
    const needle = label.toLowerCase().replace(/^[#.]/, '').trim();
    const byLabel = dom.nodes.find(
      (n) =>
        n.label?.toLowerCase() === needle ||
        n.text?.toLowerCase() === needle ||
        n.placeholder?.toLowerCase() === needle,
    );
    if (byLabel) return { node: byLabel, match: 'repaired' };
    if (needle.length >= 3) {
      const byPartial = dom.nodes.find((n) =>
        `${n.label ?? ''} ${n.text ?? ''} ${n.placeholder ?? ''}`.toLowerCase().includes(needle),
      );
      if (byPartial) return { node: byPartial, match: 'repaired' };
    }
  }

  return { match: 'none' };
}

function findNode(dom: ScrubbedDom, action: AgentAction): ScrubbedNode | undefined {
  if (!('selector' in action) || !action.selector) return undefined;
  return dom.nodes.find((n) => n.selector === action.selector);
}

function valueTokenOf(obj: Record<string, unknown>): ValueToken | undefined {
  const v = str(obj.valueType ?? obj.value_type ?? obj.valuetype);
  if (!v) return undefined;
  const upper = v.toUpperCase().replace(/[\s-]+/g, '_');
  return VALUE_TOKENS.has(upper as ValueToken) ? (upper as ValueToken) : undefined;
}

function literal(obj: Record<string, unknown>): { value?: string } {
  const value = str(obj.value ?? obj.text);
  return value ? { value } : {};
}

const str = (v: unknown): string | undefined => (typeof v === 'string' && v.trim() ? v.trim() : undefined);
const num = (v: unknown): number | undefined => {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && /^-?\d+$/.test(v.trim())) return Number(v.trim());
  return undefined;
};
const bool = (v: unknown): boolean => v === true || v === 'true';

export function extractJson(raw: string): unknown {
  // Models like to wrap output in ```json fences.
  const cleaned = raw.replace(/```(?:json)?/gi, '');
  const start = cleaned.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < cleaned.length; i++) {
    const c = cleaned[i]!;
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c === '\\') {
      escaped = true;
      continue;
    }
    if (c === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (c === '{') depth++;
    else if (c === '}') {
      depth--;
      if (depth === 0) {
        const slice = cleaned.slice(start, i + 1);
        try {
          return JSON.parse(slice);
        } catch {
          return tolerantParse(slice);
        }
      }
    }
  }
  // Unterminated object — small models truncate. Try closing it.
  return tolerantParse(`${cleaned.slice(start)}}`);
}

/** Accept single quotes and unquoted keys, which small models emit often. */
function tolerantParse(slice: string): unknown {
  const repaired = slice
    .replace(/([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:/g, '$1"$2":')
    .replace(/'/g, '"')
    .replace(/,\s*(}|$)/g, '$1');
  try {
    return JSON.parse(repaired);
  } catch {
    return null;
  }
}
