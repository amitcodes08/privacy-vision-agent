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
import { type AgentAction, type ScrubbedDom, type ScrubbedNode, type ValueToken } from '@shared/types';
import { inferValueType } from './local-planner';

/** Compact, token-cheap page description for the local model. */
export function buildPrompt(goal: string, dom: ScrubbedDom, history: AgentAction[] = []): string {
  const lines = dom.nodes.slice(0, 40).map((n) => {
    const bits = [
      n.tag,
      n.type && `type=${n.type}`,
      n.label && `label="${trunc(n.label)}"`,
      n.text && `text="${trunc(n.text)}"`,
      n.value && `value="${trunc(n.value)}"`,
      n.disabled && 'disabled',
    ]
      .filter(Boolean)
      .join(' ');
    return `${n.id}: ${bits}`;
  });
  const past = history
    .slice(-3)
    .map((h) => ('selector' in h ? `${h.action}(${h.selector})` : h.action))
    .join(', ');

  // The element *id* is what the model must return — a short integer it can
  // copy reliably. Selectors are resolved from the id on our side, which
  // removes the single largest source of unusable local output.
  return [
    'You are a browser agent. Look at the screenshot and pick the ONE next action.',
    `GOAL: ${goal}`,
    `PAGE: ${trunc(dom.title, 80)}`,
    'ELEMENTS (id: description):',
    lines.join('\n'),
    past && `ALREADY DONE: ${past}`,
    '',
    'Answer with one JSON object and nothing else.',
    'Keys: "action" (one of click, fill, scroll, done), "id" (an element id from the list above),',
    'and for fill also "valueType" (one of USER_EMAIL, USER_FULL_NAME, USER_PHONE, USER_ADDRESS, LITERAL)',
    'plus "value" when valueType is LITERAL.',
    'Use action "done" when the goal is already satisfied.',
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

/**
 * Parse the model's text into an action and score how much we trust it.
 * The score is structural, not probabilistic: a selector that does not
 * exist on the page is worthless no matter how confident the logits were.
 */
export function parseAction(raw: string, dom: ScrubbedDom): { action: AgentAction; confidence: number } {
  const json = extractJson(raw);
  if (!json || typeof json !== 'object') {
    return { action: { action: 'escalate', reason: 'no JSON in local output' }, confidence: 0 };
  }

  const { action, match } = normalize(json as Record<string, unknown>, dom);
  if (!action) return { action: { action: 'escalate', reason: 'schema mismatch' }, confidence: 0.1 };
  if (action.action === 'escalate') return { action, confidence: 0 };
  if (action.action === 'done') return { action, confidence: 0.7 };
  if (action.action === 'wait') return { action, confidence: 0.6 };
  if (action.action === 'navigate') return { action, confidence: 0.6 };
  if (action.action === 'scroll' && !action.selector) return { action, confidence: 0.65 };

  // An unresolvable selector is the one case worth escalating over: acting on
  // it would poke at an element we cannot see.
  if (match === 'unresolved') return { action, confidence: 0.12 };

  let confidence = 0.55;
  if (match === 'exact') confidence += 0.32;
  else if (match === 'repaired') confidence += 0.2;

  const node = findNode(dom, action);
  if (node?.disabled) confidence -= 0.25;
  if (action.action === 'fill' && !action.valueType) confidence -= 0.2;
  if (raw.length > 600) confidence -= 0.1;

  return { action, confidence: Math.max(0, Math.min(1, confidence)) };
}

export type Match = 'exact' | 'repaired' | 'unresolved' | 'none';

/** Coerce whatever the model produced into a valid `AgentAction`. */
function normalize(obj: Record<string, unknown>, dom: ScrubbedDom): { action: AgentAction | null; match: Match } {
  const kindRaw = String(obj.action ?? obj.type ?? obj.name ?? '').toLowerCase().trim();
  const kind = ACTION_SYNONYMS[kindRaw];
  if (!kind) return { action: null, match: 'none' };

  if (kind === 'done') {
    const summary = str(obj.summary ?? obj.reason);
    return { action: summary ? { action: 'done', summary } : { action: 'done' }, match: 'none' };
  }
  if (kind === 'escalate') {
    return { action: { action: 'escalate', reason: str(obj.reason) }, match: 'none' };
  }
  if (kind === 'wait') {
    const ms = num(obj.ms ?? obj.duration) ?? 500;
    return { action: { action: 'wait', ms: Math.max(0, Math.min(5_000, ms)) }, match: 'none' };
  }
  if (kind === 'navigate') {
    const url = str(obj.url ?? obj.href);
    if (!url || !/^https?:\/\//i.test(url)) return { action: null, match: 'none' };
    return { action: { action: 'navigate', url, reason: str(obj.reason) }, match: 'none' };
  }

  const resolved = resolveTarget(obj, dom);

  if (kind === 'scroll') {
    const deltaY = num(obj.deltaY ?? obj.delta ?? obj.amount);
    if (resolved.node) {
      return { action: { action: 'scroll', selector: resolved.node.selector, reason: str(obj.reason) }, match: resolved.match };
    }
    return {
      action: { action: 'scroll', deltaY: deltaY ?? Math.round(dom.viewport.height * 0.8), reason: str(obj.reason) },
      match: 'none',
    };
  }

  // click / fill both need a target element.
  if (!resolved.node) {
    const selector = str(obj.selector ?? obj.css ?? obj.element);
    if (!selector) return { action: null, match: 'none' };
    // Keep the model's selector but mark it unverified; the caller scores it
    // low enough that the local planner gets a shot first.
    return {
      action:
        kind === 'fill'
          ? { action: 'fill', selector, valueType: valueTokenOf(obj) ?? 'LITERAL', ...literal(obj), reason: str(obj.reason) }
          : { action: 'click', selector, reason: str(obj.reason) },
      match: 'unresolved',
    };
  }

  const node = resolved.node;
  if (kind === 'fill') {
    const valueType = valueTokenOf(obj) ?? inferValueType(node);
    return {
      action: {
        action: 'fill',
        selector: node.selector,
        valueType,
        reason: str(obj.reason),
        ...(valueType === 'LITERAL' ? literal(obj) : {}),
        ...(bool(obj.submit) ? { submit: true } : {}),
      },
      match: resolved.match,
    };
  }
  return { action: { action: 'click', selector: node.selector, reason: str(obj.reason) }, match: resolved.match };
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
