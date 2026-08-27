/**
 * Turns a live page into a compact, PII-free `ScrubbedDom`.
 *
 * Two outputs matter:
 *  1. the structural tree we may send to the cloud (already scrubbed), and
 *  2. the viewport-relative boxes the canvas redactor must black out.
 *
 * Nothing here performs network I/O — scrubbing must be verifiable offline.
 */
import {
  REDACTED_PLACEHOLDER,
  DEFAULTS,
  type BoundingBox,
  type RedactionReason,
  type ScrubbedDom,
  type ScrubbedNode,
} from '@shared/types';
import { redactText, hasPii, detectPii } from '~/privacy/pii-detector';

const INTERACTIVE = 'a,button,input,select,textarea,[role],[onclick],[contenteditable="true"],summary,label';
const SENSITIVE_AUTOCOMPLETE = new Set([
  'cc-number',
  'cc-csc',
  'cc-exp',
  'cc-exp-month',
  'cc-exp-year',
  'cc-name',
  'current-password',
  'new-password',
  'one-time-code',
]);
const SENSITIVE_NAME_RE = /pass|pwd|secret|token|cvv|cvc|card|ssn|aadhaar|otp|pin\b|account[-_]?number/i;

/** Why this element's *value* must never leave the device. */
export function classifyElement(el: Element): RedactionReason[] {
  const reasons = new Set<RedactionReason>();
  const type = (el.getAttribute('type') ?? '').toLowerCase();
  const autocomplete = (el.getAttribute('autocomplete') ?? '').toLowerCase().trim();
  const idName = `${el.getAttribute('name') ?? ''} ${el.id} ${el.getAttribute('aria-label') ?? ''}`;

  if (el.tagName === 'INPUT' && type === 'password') reasons.add('password');
  if (SENSITIVE_AUTOCOMPLETE.has(autocomplete)) {
    reasons.add(autocomplete.startsWith('cc-') ? 'credit-card' : 'autocomplete-sensitive');
  }
  if (autocomplete === 'one-time-code') reasons.add('otp');
  if (el.hasAttribute('data-pva-redact')) reasons.add('user-marked');
  if (SENSITIVE_NAME_RE.test(idName)) {
    reasons.add(/card|cvv|cvc/i.test(idName) ? 'credit-card' : 'autocomplete-sensitive');
  }
  if (el.tagName === 'INPUT' && (type === 'email' || autocomplete.includes('email'))) reasons.add('email');
  if (el.tagName === 'INPUT' && (type === 'tel' || autocomplete === 'tel')) reasons.add('phone');
  return [...reasons];
}

/** Field values are dropped entirely for these — never even a length hint. */
export function isValueForbidden(reasons: RedactionReason[]): boolean {
  return reasons.some((r) => r === 'password' || r === 'credit-card' || r === 'otp' || r === 'user-marked' || r === 'autocomplete-sensitive');
}

/** Stable, reasonably short selector the client can resolve later. */
export function cssPath(el: Element, doc: Document = el.ownerDocument!): string {
  if (el.id) {
    const idSel = `#${escapeIdent(el.id)}`;
    try {
      if (doc.querySelectorAll(idSel).length === 1) return idSel;
    } catch {
      // ignore invalid selector syntax
    }
  }
  const name = el.getAttribute('name');
  if (name) {
    const candidate = `${el.tagName.toLowerCase()}[name="${cssAttr(name)}"]`;
    try {
      if (doc.querySelectorAll(candidate).length === 1) return candidate;
    } catch {
      // ignore invalid selector syntax
    }
  }
  const parts: string[] = [];
  let node: Element | null = el;
  while (node && node.nodeType === 1 && parts.length < 6) {
    let part = node.tagName.toLowerCase();
    const parent: Element | null = node.parentElement;
    if (!parent) {
      parts.unshift(part);
      break;
    }
    const sameTag = [...parent.children].filter((c) => c.tagName === node!.tagName);
    if (sameTag.length > 1) part += `:nth-of-type(${sameTag.indexOf(node) + 1})`;
    parts.unshift(part);
    if (node.id) {
      const idSel = `#${escapeIdent(node.id)}`;
      try {
        if (doc.querySelectorAll(idSel).length === 1) {
          parts[0] = idSel;
          break;
        }
      } catch {
        // ignore invalid selector syntax
      }
    }
    node = parent;
  }
  return parts.join(' > ');
}

export const escapeIdent = (s: string): string => {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(s);
  }
  if (/^[0-9]/.test(s)) {
    return `\\3${s[0]} ${s.slice(1).replace(/([^\w-])/g, '\\$1')}`;
  }
  return s.replace(/([^\w-])/g, '\\$1');
};

const cssAttr = (s: string) => s.replace(/["\\]/g, '\\$&');

function labelFor(el: Element): string | undefined {
  const aria = el.getAttribute('aria-label');
  if (aria) return aria.trim();
  const id = el.id;
  if (id) {
    const lbl = el.ownerDocument?.querySelector(`label[for="${cssAttr(id)}"]`);
    if (lbl?.textContent) return squash(lbl.textContent);
  }
  const wrapping = el.closest('label');
  if (wrapping?.textContent) return squash(wrapping.textContent);
  const labelledBy = el.getAttribute('aria-labelledby');
  if (labelledBy) {
    const target = el.ownerDocument?.getElementById(labelledBy);
    if (target?.textContent) return squash(target.textContent);
  }
  return undefined;
}

export const squash = (s: string) => s.replace(/\s+/g, ' ').trim().slice(0, 160);

function isVisible(el: Element): boolean {
  const rect = el.getBoundingClientRect?.();
  if (!rect) return false;
  const view = el.ownerDocument?.defaultView;
  if (view?.getComputedStyle) {
    const style = view.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
  }
  // jsdom reports all-zero rects; treat that as "present but unmeasured".
  if (rect.width === 0 && rect.height === 0) return el.isConnected;
  return true;
}

function box(el: Element): BoundingBox | undefined {
  const r = el.getBoundingClientRect?.();
  if (!r || (r.width === 0 && r.height === 0)) return undefined;
  return { x: Math.round(r.x), y: Math.round(r.y), width: Math.round(r.width), height: Math.round(r.height) };
}

export interface ScrubResult {
  dom: ScrubbedDom;
  /** Viewport-relative boxes to paint black before any upload. */
  sensitiveBoxes: BoundingBox[];
  /** selector -> live element, kept client-side only. */
  index: Map<string, Element>;
}

/**
 * How many candidates we are willing to *examine* before selecting. The node
 * budget caps what we emit; this caps the cost of ranking on a huge page.
 */
const MAX_CANDIDATES = 1_500;

const OPERABLE = new Set(['a', 'button', 'input', 'select', 'textarea', 'summary']);
const OPERABLE_ROLES = new Set(['button', 'link', 'menuitem', 'tab', 'checkbox', 'radio', 'option', 'switch', 'combobox', 'textbox', 'searchbox']);

/**
 * Priority for the node budget.
 *
 * The budget used to be spent in raw document order, and `INTERACTIVE` matches
 * `[role]` — which on a real application is nearly everything. So the first 120
 * matches were the skip links, the logo, the global nav and the tab strip, and
 * the element the user was actually asking about never entered `dom.nodes` at
 * all. Neither planner can choose what it cannot see, so the VLM picked the
 * closest listed thing and the ranker keyword-matched some header link: "click
 * package.json" clicking something else entirely.
 *
 * Being *in the viewport* dominates, because the screenshot only shows the
 * viewport — an element list that disagrees with the picture is worse than a
 * short one. After that, prefer things a user can operate and things with an
 * accessible name.
 */
function priority(el: Element, view: Window | null, named: boolean): number {
  let score = 0;

  const r = el.getBoundingClientRect?.();
  const vh = view?.innerHeight ?? 0;
  const vw = view?.innerWidth ?? 0;
  if (r && r.width > 0 && r.height > 0) {
    const onScreen = r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
    if (onScreen) score += 100;
    // Just below the fold is still likely relevant; far away is not.
    else if (r.top >= vh && r.top < vh * 2) score += 30;
    if (r.width * r.height < 24) score -= 10; // tracking pixels, icon slivers
  }

  const tag = el.tagName.toLowerCase();
  const role = el.getAttribute('role');
  if (OPERABLE.has(tag)) score += 40;
  else if (role && OPERABLE_ROLES.has(role)) score += 30;
  else if (tag === 'label' || el.getAttribute('contenteditable') === 'true') score += 20;
  else if (/^h[1-3]$/.test(tag) || tag === 'legend') score += 15; // page structure
  else score += 2; // a bare [role] we do not recognise

  if (named) score += 25;
  return score;
}

export function buildScrubbedDom(
  doc: Document = document,
  maxNodes: number = DEFAULTS.maxDomNodes,
): ScrubResult {
  const view = doc.defaultView;
  const nodes: ScrubbedNode[] = [];
  const sensitiveBoxes: BoundingBox[] = [];
  const index = new Map<string, Element>();
  const summary: Partial<Record<RedactionReason, number>> = {};
  const bump = (r: RedactionReason) => {
    summary[r] = (summary[r] ?? 0) + 1;
  };

  // Two passes. First rank every visible candidate and keep the best
  // `maxNodes`; only then emit, in document order, so ids stay monotonic and
  // line up with how a reader scans the screenshot.
  const all = [...doc.querySelectorAll(INTERACTIVE), ...doc.querySelectorAll('h1,h2,h3,legend')].slice(
    0,
    MAX_CANDIDATES,
  );
  const ranked: { el: Element; order: number; score: number }[] = [];
  for (let order = 0; order < all.length; order++) {
    const el = all[order]!;
    if (!isVisible(el)) continue;
    const named = [
      el.getAttribute('aria-label'),
      el.getAttribute('placeholder'),
      el.getAttribute('name'),
      squash(directText(el)),
    ].some((s) => Boolean(s?.trim()));
    ranked.push({ el, order, score: priority(el, view, named) });
  }
  ranked.sort((a, b) => b.score - a.score || a.order - b.order);
  const selected = ranked.slice(0, maxNodes).sort((a, b) => a.order - b.order);

  let id = 0;
  for (const { el } of selected) {
    const reasons = classifyElement(el);
    const selector = cssPath(el, doc);
    index.set(selector, el);
    const rect = box(el);
    if (reasons.length > 0 && rect) sensitiveBoxes.push(rect);
    reasons.forEach(bump);

    const node: ScrubbedNode = {
      id: id++,
      tag: el.tagName.toLowerCase(),
      selector,
      visible: true,
    };
    const role = el.getAttribute('role');
    if (role) node.role = role;
    const type = el.getAttribute('type');
    if (type) node.type = type;
    const name = el.getAttribute('name');
    if (name) node.name = name;
    const label = labelFor(el);
    if (label) node.label = label;
    const placeholder = el.getAttribute('placeholder');
    if (placeholder) node.placeholder = placeholder;
    if (rect) node.box = rect;
    if (el instanceof (view?.HTMLInputElement ?? HTMLInputElement)) {
      if (el.type === 'checkbox' || el.type === 'radio') node.checked = el.checked;
    }
    if (el.hasAttribute('disabled')) node.disabled = true;
    const href = el.getAttribute('href');
    if (href) node.href = safeHref(href, doc);

    // ---- text / value scrubbing -------------------------------------
    const ownText = squash(directText(el));
    if (ownText) {
      const { text, reasons: textReasons } = redactText(ownText, REDACTED_PLACEHOLDER);
      node.text = text;
      if (textReasons.length > 0) {
        textReasons.forEach(bump);
        node.redacted = [...(node.redacted ?? []), ...textReasons];
        if (rect) sensitiveBoxes.push(rect);
      }
    }

    const rawValue = readValue(el, view);
    if (rawValue !== undefined) {
      if (isValueForbidden(reasons)) {
        node.value = REDACTED_PLACEHOLDER;
      } else if (hasPii(rawValue)) {
        const found = detectPii(rawValue).map((m) => m.reason);
        found.forEach(bump);
        node.value = REDACTED_PLACEHOLDER;
        node.redacted = [...new Set([...(node.redacted ?? []), ...found])];
        if (rect) sensitiveBoxes.push(rect);
      } else {
        node.value = squash(rawValue);
      }
    }

    if (reasons.length > 0) node.redacted = [...new Set([...(node.redacted ?? []), ...reasons])];
    nodes.push(node);
  }

  const location = doc.location ?? ({ href: '', origin: '' } as Location);
  const dom: ScrubbedDom = {
    url: stripUrl(location.href),
    origin: location.origin,
    title: squash(doc.title ?? ''),
    viewport: {
      width: view?.innerWidth ?? 0,
      height: view?.innerHeight ?? 0,
      scrollX: Math.round(view?.scrollX ?? 0),
      scrollY: Math.round(view?.scrollY ?? 0),
    },
    nodes,
    redactionSummary: summary,
  };
  return { dom, sensitiveBoxes, index };
}

/** Text belonging to this element, excluding nested interactive children. */
function directText(el: Element): string {
  let out = '';
  for (const child of el.childNodes) {
    if (child.nodeType === 3) out += child.nodeValue ?? '';
    else if (child.nodeType === 1 && !(child as Element).matches(INTERACTIVE)) {
      out += (child as Element).textContent ?? '';
    }
    if (out.length > 400) break;
  }
  return out;
}

function readValue(el: Element, view: Window | null): string | undefined {
  const W = view ?? globalThis;
  const input = (W as unknown as typeof globalThis).HTMLInputElement;
  const textarea = (W as unknown as typeof globalThis).HTMLTextAreaElement;
  const select = (W as unknown as typeof globalThis).HTMLSelectElement;
  if (input && el instanceof input) return el.type === 'checkbox' || el.type === 'radio' ? undefined : el.value;
  if (textarea && el instanceof textarea) return el.value;
  if (select && el instanceof select) return el.value;
  if (el.getAttribute('contenteditable') === 'true') return el.textContent ?? undefined;
  return undefined;
}

/** Drop query strings and fragments: they routinely carry tokens. */
export function stripUrl(href: string): string {
  try {
    const u = new URL(href);
    return `${u.origin}${u.pathname}`;
  } catch {
    return '';
  }
}

function safeHref(href: string, doc: Document): string {
  try {
    return stripUrl(new URL(href, doc.location?.href ?? 'https://localhost').href);
  } catch {
    return '';
  }
}
