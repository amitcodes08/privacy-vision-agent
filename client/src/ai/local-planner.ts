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
  taskMemory?: import('@shared/types').TaskMemory;
}

const STOPWORDS = new Set([
  'the', 'and', 'for', 'with', 'this', 'that', 'then', 'from', 'into', 'your',
  'you', 'please', 'can', 'could', 'would', 'want', 'need', 'get', 'got', 'let',
  'its', 'his', 'her', 'their', 'has', 'have', 'was', 'were', 'are', 'but',
  'not', 'all', 'any', 'out', 'off', 'now', 'page', 'site', 'website', 'button',
  'link', 'field', 'box', 'first', 'next', 'also', 'again', 'more', 'see', 'down', 'up',
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

type RankingMode = 'search' | 'target' | 'control';

const TARGET_META_WORDS = new Set([
  'find',
  'search',
  'look',
  'lookup',
  'locate',
  'get',
  'show',
  'please',
  'want',
  'need',
  'the',
  'a',
  'an',
  'for',
  'with',
  'and',
  'then',
  'add',
  'cart',
  'buy',
  'purchase',
  'open',
  'select',
  'choose',
  'pick',
  'color',
  'colour',
  'size',
  'variant',
  'version',
  'storage',
  'capacity',
  'price',
  'budget',
  'under',
  'below',
  'above',
  'over',
]);

/**
 * Generic navigation/UI noise.
 * These should almost never win when the objective is to find a target.
 */
const NAV_NOISE_RE =
  /\b(?:back\s+to\s+top|skip\s+to\s+content|search\s+this\s+page|privacy|terms(?:\s+of\s+use)?|accessibility|help|feedback|cookie(?:s)?|join\s+prime|home)\b/i;

/**
 * Generic "related item" language.
 * This prevents "iPhone ... case" from beating the actual iPhone.
 * These are not Amazon-specific; they occur across shopping/product sites.
 */
const RELATED_ITEM_RE =
  /\b(?:case|cover|sleeve|cable|charger|adapter|dock|stand|holder|mount|strap|band|screen\s+protector|protector|replacement|battery|manual|guide|skin|sticker|accessory|accessories|compatible\s+with|for\s+use\s+with)\b/i;

/**
 * Code/CSS/JS text accidentally surfaced by noisy DOMs.
 */
const CODE_LIKE_RE =
  /(?:\{[^}]*[;:][^}]*\}|#[a-z0-9_-]+\s*\{|\.?[a-z0-9_-]+\s*\{[^}]*[;:])/i;

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
  const ranking = rankCandidates(input);
  const best = ranking.candidates[0];

  if (!best) {
    if (ranking.intent.scroll) {
      return {
        action: {
          action: 'scroll',
          deltaY: Math.round(input.dom.viewport.height * 0.8),
          reason: 'scroll intent, no target element',
        },
        confidence: 0.7,
        source: 'heuristic',
      };
    }

    return {
      action: {
        action: 'invalid',
        reason: 'NO_ACTIONABLE_TARGET',
      },
      confidence: 0,
      source: 'heuristic',
    };
  }

  const act = toAction(best.node, input.goal, ranking);
  const conf = scoreToConfidence(best.score, ranking.breadth);
  let macroActions: AgentAction[] | undefined;

  if (act.action === 'fill' && act.submit) {
    macroActions = [act];
  }

  return {
    action: act,
    macroActions,
    confidence: conf,
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
  candidates: Candidate[];

  /** Number of meaningful terms used for ranking. */
  breadth: number;

  intent: Intent;

  /** What kind of problem the planner thinks this objective represents. */
  mode: RankingMode;

  /**
   * Extracted query for search/find objectives.
   * Example:
   *   "Find wireless headphones under $100"
   * becomes:
   *   "wireless headphones under $100"
   */
  searchQuery?: string;

  /**
   * Meaningful target tokens used for entity/result matching.
   */
  targetTokens: string[];
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
  const { goal, dom, history = [], taskMemory } = input;

  const targetGoal = taskMemory?.currentObjective?.trim() || goal.trim();
  const searchQuery = extractSearchQuery(targetGoal);
  const targetTokens = extractTargetTokens(searchQuery ?? targetGoal);

  const intent: Intent = {
    fill: INTENT.fill.test(targetGoal),
    click: INTENT.click.test(targetGoal),
    scroll: INTENT.scroll.test(targetGoal),
    navigate: INTENT.navigate.test(targetGoal),
  };

  const acted = new Map<string, number>();

  for (const h of history) {
    if ('selector' in h && h.selector) {
      acted.set(h.selector, (acted.get(h.selector) ?? 0) + 1);
    }
  }

  for (const selector of taskMemory?.attemptedTargets ?? []) {
    acted.set(selector, (acted.get(selector) ?? 0) + 2);
  }

  /*
   * Search mode:
   *
   * Only enter this mode when:
   *   1. the objective looks like a search/find request
   *   2. a usable search control actually exists
   *   3. we haven't already used that search control
   *
   * This prevents:
   *
   * "Find X"
   *   -> click search box
   *   -> click same search box forever
   */
  const availableSearchControl = dom.nodes.find(
    (n) =>
      !n.disabled &&
      n.visible &&
      isSearchControl(n) &&
      !hasRecentFill(history, n.selector) &&
      !n.value,
  );

  let mode: RankingMode;

  if (searchQuery && availableSearchControl) {
    mode = 'search';
  } else if (searchQuery && targetTokens.length > 0) {
    mode = 'target';
  } else {
    mode = 'control';
  }

  const candidates = dom.nodes
    .filter((n) => !n.disabled && n.visible)
    .map((node) => ({
      node,
      score: scoreNode(
        node,
        targetTokens,
        intent,
        acted,
        mode,
        searchQuery,
        targetGoal,
      ),
    }))
    .filter((c) => c.score > 0)
    .sort((a, b) => b.score - a.score);

  return {
    candidates,
    breadth: Math.max(targetTokens.length, 1),
    intent,
    mode,
    searchQuery,
    targetTokens,
  };
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

function isStateChanger(n: ScrubbedNode): boolean {
  if (n.tag === 'button') return true;
  if (n.role && ['button', 'submit', 'checkbox', 'menuitem', 'option'].includes(n.role)) return true;
  if (n.tag === 'input' && ['submit', 'button', 'reset', 'checkbox', 'radio'].includes(n.type || '')) return true;
  return false;
}

function isNavigator(n: ScrubbedNode): boolean {
  if (n.tag === 'a') return true;
  if (n.role === 'link') return true;
  return false;
}

const ACTION_VERBS = /\b(add|submit|buy|checkout|delete|remove|clear|save|update|apply|accept|agree)\b/i;
function containsVerb(text: string): boolean {
  return ACTION_VERBS.test(text);
}

const describe = (n: ScrubbedNode): string =>
  (n.label ?? n.text ?? n.placeholder ?? n.name ?? n.tag).slice(0, 40);

function scoreNode(
  n: ScrubbedNode,
  targetTokens: readonly string[],
  intent: { fill: boolean; click: boolean; scroll: boolean; navigate: boolean },
  acted: Map<string, number>,
  mode: RankingMode,
  searchQuery: string | undefined,
  targetGoal: string,
): number {
  const hay = haystack(n);
  if (!hay) return 0;

  const kind = kindOf(n);
  const repeats = acted.get(n.selector) ?? 0;

  // Never let repeated failures become increasingly attractive.
  if (repeats >= 2) return 0;

  /* ---------------------------------------------------------------
   * SEARCH MODE
   * ------------------------------------------------------------- */

  if (mode === 'search') {
    if (!isSearchControl(n)) return 0;

    let score = 20;

    if (n.role === 'searchbox') score += 8;
    if (n.type === 'search') score += 8;

    const descriptor =
      `${n.label ?? ''} ${n.placeholder ?? ''} ${n.name ?? ''}`.toLowerCase();

    if (/\bsearch\b/.test(descriptor)) score += 8;
    if (/\bquery\b/.test(descriptor)) score += 4;
    if (/\bfind\b/.test(descriptor)) score += 4;

    if (n.value && searchQuery) {
      const current = normalizeText(n.value);
      const wanted = normalizeText(searchQuery);

      /*
       * Already filled with approximately the desired query.
       * Don't fill/click it again.
       */
      if (current.includes(wanted) || wanted.includes(current)) {
        return 0;
      }
    }

    score -= repeats * 12;

    return score;
  }

  /* ---------------------------------------------------------------
   * TARGET MODE
   *
   * Example:
   *   "Find iPhone 17 Pro Max in orange 1TB"
   *
   * After the search field has been used, we stop scoring arbitrary
   * search controls and score actual candidate results.
   * ------------------------------------------------------------- */

  if (mode === 'target') {
    if (kind === 'other') return 0;

    let score = 0;

    const meaningfulTokens = targetTokens.filter(isUsefulTargetToken);
    let matched = 0;

    for (const token of meaningfulTokens) {
      if (wordMatch(hay, token)) {
        matched++;
        score += token.length >= 5 ? 5 : 3;
      }
    }

    if (meaningfulTokens.length > 0) {
      const coverage = matched / meaningfulTokens.length;
      score += coverage * 12;
    }

    /*
     * Exact/multi-word target phrase is much stronger than isolated
     * token overlap.
     */
    if (searchQuery && containsPhrase(hay, searchQuery)) {
      score += 20;
    }

    /*
     * Product/result candidates should generally be actionable.
     */
    if (kind === 'click') score += 6;
    if (n.role === 'link' || n.role === 'button') score += 3;

    /*
     * Headings inside result cards can still be useful context,
     * but should not outrank actual clickable results.
     */
    if (n.tag === 'h1' || n.tag === 'h2' || n.tag === 'h3') {
      score += 1;
    }

    /*
     * Kill common page chrome / navigation noise.
     */
    if (NAV_NOISE_RE.test(hay)) {
      score -= 16;
    }

    /*
     * A product-related objective should not pick:
     *   iPhone 17 Pro Max CASE
     * over
     *   iPhone 17 Pro Max
     */
    if (
      matched >= Math.min(2, meaningfulTokens.length) &&
      RELATED_ITEM_RE.test(hay)
    ) {
      score -= 24;
    }

    /*
     * CSS/JS accidentally included in DOM.
     * Example from your log:
     * ".apex-savings-percent {font-weight: 300;"
     */
    if (CODE_LIKE_RE.test(hay)) {
      score -= 30;
    }

    /*
     * Sponsored/promoted content is not automatically rejected:
     * a sponsored result can still be the correct target.
     * We only penalize it.
     */
    if (/\b(?:sponsored|promoted|advertisement|advertising)\b/i.test(hay)) {
      score -= matched >= 2 ? 4 : 10;
    }

    score -= repeats * 12;

    return Math.max(0, score);
  }

  /* ---------------------------------------------------------------
   * CONTROL MODE
   *
   * Example:
   *   "Click Add to cart"
   *   "Select orange"
   *   "Open checkout"
   *   "Close the popup"
   * ------------------------------------------------------------- */

  let keyword = 0;

  const wanted = expand(tokenize(targetGoal));

  for (const w of wanted) {
    if (!w) continue;

    if (wordMatch(hay, w)) {
      keyword += w.length >= 4 ? w.length : 2;
    } else if (w.length >= 4 && hay.includes(w)) {
      keyword += w.length / 2;
    }
  }

  if (keyword < 2) return 0;

  let score = keyword;

  const isChanger = isStateChanger(n);
  const isNav = isNavigator(n);
  const hasVerb = containsVerb(hay);

  if (intent.fill && kind === 'fill') score += 8;
  if (intent.fill && kind === 'click') score -= 5;

  if (intent.click) {
    if (isChanger) {
      score += 8;
    } else if (isNav) {
      if (hasVerb) score += 4;
      else score -= 6;
    } else if (kind === 'click') {
      score += 5;
    }
  } else if (intent.navigate) {
    if (isNav) score += 6;
    else if (isChanger) score -= 3;
  }

  if (kind === 'other') score -= 8;

  if (n.role === 'button' || n.role === 'link' || n.role === 'menuitem') {
    score += 2;
  }

  if (NAV_NOISE_RE.test(hay)) {
    score -= 18;
  }

  if (CODE_LIKE_RE.test(hay)) {
    score -= 30;
  }

  score -= repeats * 12;

  if (kind === 'fill' && n.value) {
    score -= 4;
  }

  return Math.max(0, score);
}

function toAction(
  node: ScrubbedNode,
  goal: string,
  ranking: Ranking,
): AgentAction {
  const kind = kindOf(node);

  /*
   * Universal search behavior:
   *
   * Find/search/look for X
   *      ↓
   * search input
   *      ↓
   * fill X
   *      ↓
   * submit
   */
  if (ranking.mode === 'search' && kind === 'fill') {
    const query = ranking.searchQuery ?? literalFor(goal);

    if (query) {
      return {
        action: 'fill',
        selector: node.selector,
        valueType: 'LITERAL',
        value: query,
        submit: true,
        reason: `search objective "${query}"`,
      };
    }

    return {
      action: 'click',
      selector: node.selector,
      reason: `focus search control "${describe(node)}"`,
    };
  }

  if (kind === 'fill') {
    const valueType = inferValueType(node);

    if (valueType !== 'LITERAL') {
      return {
        action: 'fill',
        selector: node.selector,
        valueType,
        reason: `keyword match on "${describe(node)}"`,
      };
    }

    const literal = literalFor(goal);

    if (!literal) {
      return {
        action: 'click',
        selector: node.selector,
        reason: `focus "${describe(node)}"`,
      };
    }

    return {
      action: 'fill',
      selector: node.selector,
      valueType,
      value: literal,
      submit: /\b(search|find|submit|send|go|query)\b/i.test(goal),
      reason: `keyword match on "${describe(node)}"`,
    };
  }

  return {
    action: 'click',
    selector: node.selector,
    reason: `${ranking.mode} match on "${describe(node)}"`,
  };
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
export function literalFor(goal: string): string | undefined {
  const quoted = goal.match(/["“']([^"”']{2,120})["”']/);
  if (quoted?.[1]) {
    return quoted[1].trim();
  }

  const query = extractSearchQuery(goal);
  if (query) return query.replace(/^["'“](.+)["'”]$/, '$1').trim();

  const after = goal.match(
    /\b(?:search(?:\s+for)?|type|enter|fill(?:\s+in)?|write|query|look\s+for)\b[:\s]+(.{2,120})$/i,
  );

  if (after?.[1]) {
    return after[1]
      .replace(/\s+(?:and|then)\s+(?:add|buy|purchase|open|select|choose|click|submit)\b.*$/i, '')
      .replace(/^["'“](.+)["'”]$/, '$1')
      .trim();
  }

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
  return [n.label, n.text, n.context, n.placeholder, n.name, n.role, n.type, n.href?.split('/').slice(3).join(' ')]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
}



export function tokenize(goal: string): string[] {
  return (goal.toLowerCase().match(/[a-z0-9][a-z0-9-]{0,}/g) ?? []).filter((w) => w.length >= 2 && !STOPWORDS.has(w));
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

function extractSearchQuery(goal: string): string | undefined {
  const match = goal.match(
    /^\s*(?:find(?:\s+me)?|search(?:\s+for)?|look\s+for|lookup|look\s+up|locate|show\s+me|get\s+me)\s+(.+)$/i,
  );

  if (!match?.[1]) return undefined;

  let query = match[1].trim();

  /*
   * Stop before a later task instruction.
   *
   * "Find wireless headphones and add one to cart"
   * →
   * "wireless headphones"
   */
  query = query.replace(
    /\s+(?:and|then)\s+(?:add|buy|purchase|open|select|choose|click|submit|checkout|book|reserve|save)\b.*$/i,
    '',
  );

  query = query.replace(/^["'“](.+)["'”]$/, '$1');
  query = query.replace(/[.!?,;:]+$/, '').trim();
  query = query.replace(/^["'“](.+)["'”]$/, '$1').trim();

  return query.length >= 2 ? query : undefined;
}

function extractTargetTokens(text: string): string[] {
  const tokens = tokenize(text);

  return tokens.filter(
    (token) => !TARGET_META_WORDS.has(token),
  );
}

function isUsefulTargetToken(token: string): boolean {
  /*
   * Bare small numbers are dangerous in noisy DOMs.
   *
   * "17" matched "17.42 cm" in your Amazon trace.
   *
   * We still keep:
   *   1tb
   *   128gb
   *   gen2
   *   x5
   */
  if (/^\d+$/.test(token) && token.length < 4) {
    return false;
  }

  return token.length >= 3;
}

function isSearchControl(n: ScrubbedNode): boolean {
  if (n.tag !== 'input' && n.role !== 'searchbox' && n.role !== 'combobox') {
    return false;
  }

  if (n.type === 'search') return true;
  if (n.role === 'searchbox') return true;

  const descriptor =
    `${n.name ?? ''} ${n.label ?? ''} ${n.placeholder ?? ''}`.toLowerCase();

  return /\b(?:search|find|query|lookup)\b/.test(descriptor);
}

function hasRecentFill(
  history: readonly AgentAction[],
  selector: string,
): boolean {
  return history.some(
    (action) =>
      action.action === 'fill' &&
      'selector' in action &&
      action.selector === selector,
  );
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function containsPhrase(hay: string, phrase: string): boolean {
  const h = normalizeText(hay);
  const p = normalizeText(phrase);

  if (!p) return false;

  return h.includes(p);
}
