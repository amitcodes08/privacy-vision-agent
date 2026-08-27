/**
 * Deterministic, offline planner that runs *on the client*.
 *
 * Why this exists: the local VLM is small and will often be unsure, and the
 * escalation server is a network round trip away. Without a local fallback,
 * "unsure" meant "upload a frame", so effectively every step was escalating.
 * This planner sits between the two — it is dumb but instant, private, and
 * right often enough that the server becomes a genuine last resort.
 *
 * No imports beyond the shared contract: this must stay usable from the
 * service worker without pulling in onnxruntime.
 */
import type { AgentAction, AgentDecision, ScrubbedDom, ScrubbedNode } from '@shared/types';

export interface PlanInput {
  goal: string;
  dom: ScrubbedDom;
  history?: AgentAction[];
}

const STOPWORDS = new Set([
  'the', 'and', 'for', 'with', 'this', 'that', 'then', 'from', 'into', 'your',
  'you', 'please', 'can', 'could', 'would', 'want', 'need', 'get', 'got', 'let',
  'its', 'his', 'her', 'their', 'has', 'have', 'was', 'were', 'are', 'but',
  'not', 'all', 'any', 'out', 'off', 'now', 'page', 'site', 'website', 'button',
  'link', 'field', 'box', 'first', 'next', 'also', 'again',
  // Pure operation verbs. These are already read as *intent* below, and as
  // content keywords they are noise: "click on package.json" scored every
  // element for the word "click". Words that double as real labels — accept,
  // login, search, submit, close — deliberately stay out of this list.
  'click', 'clicks', 'clicking', 'press', 'tap', 'hit', 'push', 'type', 'paste',
  'navigate', 'visit', 'browse', 'goto',
]);

/** Verbs that tell us which *kind* of element to prefer. */
const INTENT = {
  fill: /\b(fill|type|enter|input|write|put|set|paste|search\s+for|query)\b/i,
  click: /\b(click|press|tap|hit|push|open|choose|select|pick|accept|agree|allow|dismiss|close|submit|send|login|log\s?in|sign\s?in|sign\s?up|register|continue|proceed|confirm|checkout|add|apply|follow|expand|toggle)\b/i,
  scroll: /\b(scroll|scroll\s?down|scroll\s?up|page\s?down|bottom|top|load\s?more)\b/i,
  navigate: /\b(go\s?to|navigate|visit|browse)\b/i,
} as const;

/**
 * Words that mean the same thing to a user but rarely appear verbatim on the
 * element they are looking for. Expanding these is most of why this planner
 * beats a plain substring match.
 */
const SYNONYMS: Record<string, readonly string[]> = {
  accept: ['accept', 'agree', 'allow', 'ok', 'okay', 'got it', 'understood', 'consent', 'yes'],
  cookies: ['cookie', 'cookies', 'consent', 'gdpr', 'privacy', 'tracking'],
  login: ['login', 'log in', 'signin', 'sign in', 'log on', 'authenticate'],
  signup: ['signup', 'sign up', 'register', 'create account', 'join', 'get started'],
  logout: ['logout', 'log out', 'signout', 'sign out'],
  search: ['search', 'find', 'query', 'lookup', 'q'],
  submit: ['submit', 'send', 'continue', 'next', 'proceed', 'confirm', 'go'],
  cart: ['cart', 'basket', 'bag', 'checkout'],
  settings: ['settings', 'preferences', 'options', 'account', 'profile', 'config'],
  billing: ['billing', 'payment', 'invoice', 'subscription', 'plan', 'pricing'],
  close: ['close', 'dismiss', 'cancel', 'no thanks', 'not now', 'skip', 'later'],
  menu: ['menu', 'navigation', 'nav', 'hamburger', 'more'],
  email: ['email', 'e-mail', 'mail', 'username'],
  password: ['password', 'passwd', 'pwd'],
  name: ['name', 'full name', 'fullname', 'firstname', 'first name'],
  phone: ['phone', 'tel', 'telephone', 'mobile', 'number'],
  address: ['address', 'street', 'city', 'zip', 'postal'],
};

const CLICKABLE = new Set(['a', 'button', 'summary', 'label']);
const FILLABLE = new Set(['input', 'textarea']);
const NON_TEXT_INPUT = new Set(['checkbox', 'radio', 'file', 'range', 'color', 'submit', 'button', 'reset', 'image', 'hidden']);

export function planLocally(input: PlanInput): AgentDecision {
  const { goal, dom, history = [] } = input;
  const ranking = rankCandidates(input);
  const best = ranking.candidates[0];

  // Nothing matched. A bare "scroll" goal still has an answer; otherwise the
  // honest outcome is to stop rather than poke at a random element.
  if (!best) {
    if (ranking.intent.scroll) {
      return {
        action: { action: 'scroll', deltaY: Math.round(dom.viewport.height * 0.8), reason: 'scroll intent, no target element' },
        confidence: 0.7,
        source: 'heuristic',
      };
    }
    return {
      action: {
        action: 'done',
        summary: history.length
          ? 'no remaining element matches the goal'
          : 'no element on this page matches the goal',
      },
      confidence: history.length ? 0.55 : 0.3,
      source: 'heuristic',
    };
  }

  return {
    action: toAction(best.node, goal),
    confidence: scoreToConfidence(best.score, ranking.breadth),
    source: 'heuristic',
  };
}

export interface Candidate {
  node: ScrubbedNode;
  score: number;
}

export interface Intent {
  fill: boolean;
  click: boolean;
  scroll: boolean;
  navigate: boolean;
}

export interface Ranking {
  /** Goal-relevant nodes, most relevant first. Empty when nothing matches. */
  candidates: Candidate[];
  /** How many expanded goal keywords were searched for; normalises scores. */
  breadth: number;
  intent: Intent;
}

/**
 * Score every node against the goal.
 *
 * Used for two different jobs, which is the point: it picks which elements go
 * into the VLM's prompt (so the model can actually see the one it needs), and
 * it independently corroborates whatever the VLM then chooses. The VLM does
 * the planning; this only decides what it gets to look at.
 */
export function rankCandidates(input: PlanInput): Ranking {
  const { goal, dom, history = [] } = input;
  const wanted = expand(tokenize(goal));

  const intent: Intent = {
    fill: INTENT.fill.test(goal),
    click: INTENT.click.test(goal),
    scroll: INTENT.scroll.test(goal),
    navigate: INTENT.navigate.test(goal),
  };

  // A selector we already acted on twice is a loop; stop preferring it.
  const acted = new Map<string, number>();
  for (const h of history) {
    const sel = 'selector' in h ? h.selector : undefined;
    if (sel) acted.set(sel, (acted.get(sel) ?? 0) + 1);
  }

  const candidates = dom.nodes
    .filter((n) => !n.disabled && n.visible)
    .map((node) => ({ node, score: scoreNode(node, wanted, intent, acted) }))
    .filter((c) => c.score > 0)
    .sort((a, b) => b.score - a.score);

  return { candidates, breadth: wanted.length, intent };
}

/**
 * Where the ranker placed a selector, or undefined if it did not rank it at
 * all. `0` means "this is what I would have picked too".
 */
export function rankOf(ranking: Ranking, selector: string): number | undefined {
  const at = ranking.candidates.findIndex((c) => c.node.selector === selector);
  return at === -1 ? undefined : at;
}

/** A strong keyword match is as good as this planner ever gets — cap it. */
function scoreToConfidence(score: number, breadth: number): number {
  return Math.min(0.78, 0.34 + score / (6 + breadth * 1.4));
}

function scoreNode(
  n: ScrubbedNode,
  wanted: readonly string[],
  intent: { fill: boolean; click: boolean; scroll: boolean; navigate: boolean },
  acted: Map<string, number>,
): number {
  const hay = haystack(n);
  if (!hay) return 0;

  let keyword = 0;
  for (const w of wanted) {
    if (!w) continue;
    if (wordMatch(hay, w)) keyword += w.length >= 4 ? w.length : 2;
    // Partial credit for long tokens only. A 3-letter substring match is
    // noise — it is what made "log in" fire on a "Blog" link.
    else if (w.length >= 4 && hay.includes(w)) keyword += w.length / 2;
  }

  // The intent bonus amplifies a real keyword match; it must never create one,
  // or "click sign in" would click an arbitrary button on a page without one.
  if (keyword < 2) return 0;

  let score = keyword;
  const kind = kindOf(n);

  // Intent alignment. Deliberately gentle — a keyword match on the wrong kind
  // of element is still better than escalating.
  if (intent.fill && kind === 'fill') score += 6;
  if (intent.fill && kind === 'click') score -= 2;
  if (intent.click && kind === 'click') score += 6;
  if (intent.click && kind === 'fill') score -= 2;
  if (!intent.fill && !intent.click && kind !== 'other') score += 2;

  // Prefer things a user can actually operate.
  if (kind === 'other') score -= 3;
  if (n.role === 'button' || n.role === 'link' || n.role === 'menuitem') score += 2;

  // Loop guard: acting on the same selector repeatedly is never the plan.
  const repeats = acted.get(n.selector) ?? 0;
  score -= repeats * 9;

  // A filled field usually does not need filling again.
  if (kind === 'fill' && n.value && n.value.length > 0) score -= 3;

  return score;
}

function toAction(node: ScrubbedNode, goal: string): AgentAction {
  const kind = kindOf(node);

  if (kind === 'fill') {
    const valueType = inferValueType(node);
    if (valueType !== 'LITERAL') {
      return { action: 'fill', selector: node.selector, valueType, reason: `keyword match on "${describe(node)}"` };
    }
    const literal = literalFor(goal);
    // Without text to type, a literal fill would clear the field for nothing;
    // focusing it is the honest degradation.
    if (!literal) {
      return { action: 'click', selector: node.selector, reason: `focus "${describe(node)}"` };
    }
    return {
      action: 'fill',
      selector: node.selector,
      valueType,
      value: literal,
      submit: /\b(search|find|submit|send|go)\b/i.test(goal),
      reason: `keyword match on "${describe(node)}"`,
    };
  }

  return { action: 'click', selector: node.selector, reason: `keyword match on "${describe(node)}"` };
}

/** Which private value the client should hydrate into this field. */
export function inferValueType(n: ScrubbedNode): 'USER_EMAIL' | 'USER_FULL_NAME' | 'USER_PHONE' | 'USER_ADDRESS' | 'LITERAL' {
  const hay = `${n.type ?? ''} ${n.name ?? ''} ${n.label ?? ''} ${n.placeholder ?? ''}`.toLowerCase();
  if (n.type === 'email' || /\be-?mail\b/.test(hay)) return 'USER_EMAIL';
  if (n.type === 'tel' || /\b(phone|tel|mobile)\b/.test(hay)) return 'USER_PHONE';
  if (/\b(address|street|city|zip|postal)\b/.test(hay)) return 'USER_ADDRESS';
  if (/\b(full\s?name|fullname|your\s?name|first\s?name|last\s?name)\b/.test(hay)) return 'USER_FULL_NAME';
  return 'LITERAL';
}

/**
 * Pull the quoted or trailing phrase out of the goal for a literal fill:
 * `search for "wireless mouse"` -> `wireless mouse`.
 */
function literalFor(goal: string): string | undefined {
  const quoted = goal.match(/["“']([^"”']{2,80})["”']/);
  if (quoted?.[1]) return quoted[1].trim();
  const after = goal.match(/\b(?:search(?:\s+for)?|type|enter|fill(?:\s+in)?|write|query|look\s+for)\b[:\s]+(.{2,80})$/i);
  if (after?.[1]) return after[1].replace(/\b(in|into|to)\s+the\s+.*$/i, '').trim() || undefined;
  return undefined;
}

type Kind = 'click' | 'fill' | 'other';

function kindOf(n: ScrubbedNode): Kind {
  if (FILLABLE.has(n.tag)) {
    if (n.tag === 'input' && n.type && NON_TEXT_INPUT.has(n.type)) return 'click';
    return 'fill';
  }
  if (CLICKABLE.has(n.tag) || n.tag === 'select') return 'click';
  if (n.role && ['button', 'link', 'menuitem', 'tab', 'checkbox', 'option'].includes(n.role)) return 'click';
  return 'other';
}

function haystack(n: ScrubbedNode): string {
  return [n.label, n.text, n.placeholder, n.name, n.role, n.type, n.href?.split('/').slice(3).join(' ')]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
}

const describe = (n: ScrubbedNode): string =>
  (n.label ?? n.text ?? n.placeholder ?? n.name ?? n.tag).slice(0, 40);

export function tokenize(goal: string): string[] {
  return (goal.toLowerCase().match(/[a-z][a-z0-9-]{1,}/g) ?? []).filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

/** Goal tokens plus their synonym groups, de-duplicated. */
function expand(tokens: readonly string[]): string[] {
  const out = new Set<string>(tokens);
  for (const t of tokens) {
    for (const [key, group] of Object.entries(SYNONYMS)) {
      if (key === t || group.includes(t)) group.forEach((g) => out.add(g));
    }
  }
  return [...out];
}

/** Whole-word-ish match, so "log" does not fire on "blog". */
function wordMatch(hay: string, needle: string): boolean {
  const at = hay.indexOf(needle);
  if (at === -1) return false;
  const before = at === 0 ? ' ' : hay[at - 1]!;
  const after = at + needle.length >= hay.length ? ' ' : hay[at + needle.length]!;
  return !/[a-z0-9]/.test(before) && !/[a-z0-9]/.test(after);
}
