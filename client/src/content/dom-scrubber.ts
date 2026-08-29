/**
 * Turns a live page into a compact, PII-free `ScrubbedDom`.
 *
 * Responsibilities:
 *  - Build the scrubbed DOM representation.
 *  - Detect sensitive values and produce redaction boxes.
 *  - Preserve parent/child/depth relationships between emitted nodes.
 *
 * IMPORTANT:
 *  - No network I/O happens here.
 *  - Sensitive values are scrubbed before the DOM representation is returned.
 *  - parentId / childIds only refer to nodes that survived the node budget.
 */

import {
  REDACTED_PLACEHOLDER,
  DEFAULTS,
  type BoundingBox,
  type RedactionReason,
  type ScrubbedDom,
  type ScrubbedNode,
} from '@shared/types';

import {
  redactText,
  hasPii,
  detectPii,
} from '~/privacy/pii-detector';

/* ------------------------------------------------------------------ *
 * Constants
 * ------------------------------------------------------------------ */

const INTERACTIVE =
  'a,button,input,select,textarea,[role],[onclick],[contenteditable="true"],summary,label';

const SENSITIVE_AUTOCOMPLETE = new Set<string>([
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

const SENSITIVE_NAME_RE =
  /pass|pwd|secret|token|cvv|cvc|card|ssn|aadhaar|otp|pin\b|account[-_]?number/i;

const MAX_CANDIDATES = 1_500;

const OPERABLE = new Set<string>([
  'a',
  'button',
  'input',
  'select',
  'textarea',
  'summary',
]);

const OPERABLE_ROLES = new Set<string>([
  'button',
  'link',
  'menuitem',
  'tab',
  'checkbox',
  'radio',
  'option',
  'switch',
  'combobox',
  'textbox',
  'searchbox',
]);

/* ------------------------------------------------------------------ *
 * Privacy classification
 * ------------------------------------------------------------------ */

/**
 * Why this element's value/text must be protected.
 */
export function classifyElement(
  el: Element,
): RedactionReason[] {
  const reasons = new Set<RedactionReason>();

  const type =
    (el.getAttribute('type') ?? '').toLowerCase();

  const autocomplete =
    (el.getAttribute('autocomplete') ?? '')
      .toLowerCase()
      .trim();

  const idName = [
    el.getAttribute('name') ?? '',
    el.id,
    el.getAttribute('aria-label') ?? '',
  ].join(' ');

  if (
    el.tagName === 'INPUT' &&
    type === 'password'
  ) {
    reasons.add('password');
  }

  if (
    SENSITIVE_AUTOCOMPLETE.has(autocomplete)
  ) {
    reasons.add(
      autocomplete.startsWith('cc-')
        ? 'credit-card'
        : 'autocomplete-sensitive',
    );
  }

  if (
    autocomplete === 'one-time-code'
  ) {
    reasons.add('otp');
  }

  if (
    el.hasAttribute('data-pva-redact')
  ) {
    reasons.add('user-marked');
  }

  if (
    SENSITIVE_NAME_RE.test(idName)
  ) {
    reasons.add(
      /card|cvv|cvc/i.test(idName)
        ? 'credit-card'
        : 'autocomplete-sensitive',
    );
  }

  if (
    el.tagName === 'INPUT' &&
    (
      type === 'email' ||
      autocomplete.includes('email')
    )
  ) {
    reasons.add('email');
  }

  if (
    el.tagName === 'INPUT' &&
    (
      type === 'tel' ||
      autocomplete === 'tel'
    )
  ) {
    reasons.add('phone');
  }

  return [...reasons];
}

/**
 * These values must never leave the browser.
 */
export function isValueForbidden(
  reasons: RedactionReason[],
): boolean {
  return reasons.some(
    (reason) =>
      reason === 'password' ||
      reason === 'credit-card' ||
      reason === 'otp' ||
      reason === 'user-marked' ||
      reason === 'autocomplete-sensitive',
  );
}

/* ------------------------------------------------------------------ *
 * Selector generation
 * ------------------------------------------------------------------ */

export function cssPath(
  el: Element,
  doc: Document = el.ownerDocument!,
): string {
  if (el.id) {
    const idSel =
      `#${escapeIdent(el.id)}`;

    try {
      if (
        doc.querySelectorAll(idSel).length === 1
      ) {
        return idSel;
      }
    } catch {
      // Ignore invalid selector syntax.
    }
  }

  const name =
    el.getAttribute('name');

  if (name) {
    const candidate =
      `${el.tagName.toLowerCase()}[name="${cssAttr(name)}"]`;

    try {
      if (
        doc.querySelectorAll(candidate).length === 1
      ) {
        return candidate;
      }
    } catch {
      // Ignore invalid selector syntax.
    }
  }

  const parts: string[] = [];
  let node: Element | null = el;

  while (
    node &&
    node.nodeType === 1 &&
    parts.length < 6
  ) {
    let part =
      node.tagName.toLowerCase();

    const parentElement: Element | null =
      node.parentElement;

    if (!parentElement) {
      parts.unshift(part);
      break;
    }

    const sameTag = [
      ...parentElement.children,
    ].filter(
      (child) =>
        child.tagName === node!.tagName,
    );

    if (sameTag.length > 1) {
      const index =
        sameTag.indexOf(node);

      part +=
        `:nth-of-type(${index + 1})`;
    }

    parts.unshift(part);

    if (node.id) {
      const idSel =
        `#${escapeIdent(node.id)}`;

      try {
        if (
          doc.querySelectorAll(idSel).length === 1
        ) {
          parts[0] = idSel;
          break;
        }
      } catch {
        // Ignore invalid selector syntax.
      }
    }

    node = parentElement;
  }

  return parts.join(' > ');
}

export const escapeIdent = (
  s: string,
): string => {
  if (
    typeof CSS !== 'undefined' &&
    typeof CSS.escape === 'function'
  ) {
    return CSS.escape(s);
  }

  if (/^[0-9]/.test(s)) {
    return `\\3${s[0]} ${s
      .slice(1)
      .replace(
        /([^\w-])/g,
        '\\$1',
      )}`;
  }

  return s.replace(
    /([^\w-])/g,
    '\\$1',
  );
};

const cssAttr = (
  s: string,
): string =>
  s.replace(
    /["\\]/g,
    '\\$&',
  );

/* ------------------------------------------------------------------ *
 * Accessibility / text helpers
 * ------------------------------------------------------------------ */

function labelFor(
  el: Element,
): string | undefined {
  const aria =
    el.getAttribute('aria-label');

  if (aria) {
    return aria.trim();
  }

  const id =
    el.id;

  if (id) {
    try {
      const label =
        el.ownerDocument?.querySelector(
          `label[for="${cssAttr(id)}"]`,
        );

      if (label?.textContent) {
        return squash(
          label.textContent,
        );
      }
    } catch {
      // Ignore malformed selector.
    }
  }

  const wrapping =
    el.closest('label');

  if (wrapping?.textContent) {
    return squash(
      wrapping.textContent,
    );
  }

  const labelledBy =
    el.getAttribute(
      'aria-labelledby',
    );

  if (labelledBy) {
    const text =
      labelledBy
        .split(/\s+/)
        .filter(Boolean)
        .map((idRef) => {
          const target =
            el.ownerDocument?.getElementById(
              idRef,
            );

          return target?.textContent ?? '';
        })
        .join(' ');

    if (text.trim()) {
      return squash(text);
    }
  }

  return undefined;
}

export const squash = (
  s: string,
): string =>
  s
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 160);

/* ------------------------------------------------------------------ *
 * Visibility / geometry
 * ------------------------------------------------------------------ */

function isVisible(
  el: Element,
): boolean {
  const rect =
    el.getBoundingClientRect?.();

  if (!rect) {
    return false;
  }

  const view =
    el.ownerDocument?.defaultView;

  if (view?.getComputedStyle) {
    const style =
      view.getComputedStyle(el);

    if (
      style.display === 'none' ||
      style.visibility === 'hidden' ||
      style.opacity === '0'
    ) {
      return false;
    }
  }

  /*
   * jsdom normally reports zero-sized rectangles.
   * Treat connected elements as present during tests.
   */
  if (
    rect.width === 0 &&
    rect.height === 0
  ) {
    return el.isConnected;
  }

  return true;
}

function box(
  el: Element,
): BoundingBox | undefined {
  const rect =
    el.getBoundingClientRect?.();

  if (
    !rect ||
    (
      rect.width === 0 &&
      rect.height === 0
    )
  ) {
    return undefined;
  }

  return {
    x: Math.round(rect.x),
    y: Math.round(rect.y),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
  };
}

/* ------------------------------------------------------------------ *
 * Scrape result
 * ------------------------------------------------------------------ */

export interface ScrubResult {
  dom: ScrubbedDom;

  /**
   * Viewport-relative boxes that must be redacted before upload.
   */
  sensitiveBoxes: BoundingBox[];

  /**
   * selector -> live element.
   *
   * This never leaves the content script.
   */
  index: Map<string, Element>;
}

/* ------------------------------------------------------------------ *
 * DOM prioritisation
 * ------------------------------------------------------------------ */

function priority(
  el: Element,
  view: Window | null,
  named: boolean,
): number {
  let score = 0;

  const rect =
    el.getBoundingClientRect?.();

  const vh =
    view?.innerHeight ?? 0;

  const vw =
    view?.innerWidth ?? 0;

  if (
    rect &&
    rect.width > 0 &&
    rect.height > 0
  ) {
    const onScreen =
      rect.bottom > 0 &&
      rect.top < vh &&
      rect.right > 0 &&
      rect.left < vw;

    if (onScreen) {
      score += 100;
    } else if (
      rect.top >= vh &&
      rect.top < vh * 2
    ) {
      score += 30;
    }

    if (
      rect.width * rect.height <
      24
    ) {
      score -= 10;
    }
  }

  const tag =
    el.tagName.toLowerCase();

  const role =
    el.getAttribute('role');

  if (OPERABLE.has(tag)) {
    score += 40;
  } else if (
    role &&
    OPERABLE_ROLES.has(role)
  ) {
    score += 30;
  } else if (
    tag === 'label' ||
    el.getAttribute(
      'contenteditable',
    ) === 'true'
  ) {
    score += 20;
  } else if (
    /^h[1-3]$/.test(tag) ||
    tag === 'legend'
  ) {
    score += 15;
  } else {
    score += 2;
  }

  if (named) {
    score += 25;
  }

  return score;
}

/* ------------------------------------------------------------------ *
 * Main scrubber
 * ------------------------------------------------------------------ */

export function buildScrubbedDom(
  doc: Document = document,
  maxNodes: number =
    DEFAULTS.maxDomNodes,
): ScrubResult {
  const view =
    doc.defaultView;

  const nodes: ScrubbedNode[] =
    [];

  const sensitiveBoxes:
    BoundingBox[] = [];

  const index =
    new Map<string, Element>();

  const formMap = new Map<Element, number>();
  let nextFormId = 1;

  const summary:
    Partial<Record<
      RedactionReason,
      number
    >> = {};

  const bump = (
    reason: RedactionReason,
  ): void => {
    summary[reason] =
      (summary[reason] ?? 0) + 1;
  };

  /* -------------------------------------------------------------- *
   * Gather candidate elements
   * -------------------------------------------------------------- */

  const all = [
    ...doc.querySelectorAll(
      INTERACTIVE,
    ),
    ...doc.querySelectorAll(
      'h1,h2,h3,legend',
    ),
  ].slice(
    0,
    MAX_CANDIDATES,
  );

  /* -------------------------------------------------------------- *
   * Rank candidates
   * -------------------------------------------------------------- */

  const ranked: {
    el: Element;
    order: number;
    score: number;
  }[] = [];

  for (
    let order = 0;
    order < all.length;
    order++
  ) {
    const el =
      all[order]!;

    if (!isVisible(el)) {
      continue;
    }

    const named = [
      el.getAttribute(
        'aria-label',
      ),
      el.getAttribute(
        'placeholder',
      ),
      el.getAttribute(
        'name',
      ),
      squash(
        directText(el),
      ),
    ].some(
      (value) =>
        Boolean(
          value?.trim(),
        ),
    );

    ranked.push({
      el,
      order,
      score:
        priority(
          el,
          view,
          named,
        ),
    });
  }

  ranked.sort(
    (a, b) =>
      b.score - a.score ||
      a.order - b.order,
  );

  /*
   * Select the strongest nodes, then restore DOM order.
   *
   * This is important because the numeric IDs need to correspond to
   * the visual/document structure rather than arbitrary ranking order.
   */
  const selected =
    ranked
      .slice(0, maxNodes)
      .sort(
        (a, b) =>
          a.order - b.order,
      );

  /* -------------------------------------------------------------- *
   * Map live DOM elements → emitted numeric IDs
   * -------------------------------------------------------------- */

  const selectedId =
    new Map<Element, number>();

  for (
    let i = 0;
    i < selected.length;
    i++
  ) {
    selectedId.set(
      selected[i]!.el,
      i,
    );
  }

  /* -------------------------------------------------------------- *
   * Build scrubbed nodes
   * -------------------------------------------------------------- */

  for (
    let nodeId = 0;
    nodeId < selected.length;
    nodeId++
  ) {
    const el =
      selected[nodeId]!.el;

    const reasons =
      classifyElement(el);

    const selector =
      cssPath(el, doc);

    index.set(
      selector,
      el,
    );

    const rect =
      box(el);

    if (
      reasons.length > 0 &&
      rect
    ) {
      sensitiveBoxes.push(
        rect,
      );
    }

    reasons.forEach(
      bump,
    );

    /* ------------------------------------------------------------ *
     * Find nearest emitted parent
     * ------------------------------------------------------------ */

    let parentId:
      | number
      | undefined;

    let parentElement:
      | Element
      | null =
      el.parentElement;

    while (parentElement) {
      const candidateId =
        selectedId.get(
          parentElement,
        );

      if (
        candidateId !==
        undefined
      ) {
        parentId =
          candidateId;

        break;
      }

      parentElement =
        parentElement.parentElement;
    }

    /* ------------------------------------------------------------ *
     * Calculate depth
     * ------------------------------------------------------------ */

    let depth = 0;

    if (
      parentId !== undefined
    ) {
      const parentNode =
        nodes[parentId];

      depth =
        (parentNode?.depth ?? 0) +
        1;
    }

    let formId: number | undefined;
    const formEl = (el as HTMLInputElement).form ?? el.closest('form');
    if (formEl) {
      if (!formMap.has(formEl)) {
        formMap.set(formEl, nextFormId++);
      }
      formId = formMap.get(formEl);
    }

    /* ------------------------------------------------------------ *
     * Base node
     * ------------------------------------------------------------ */

    const node: ScrubbedNode = {
      id: nodeId,

      tag:
        el.tagName.toLowerCase(),

      selector,

      visible: true,

      ...(parentId !== undefined
        ? { parentId }
        : {}),

      ...(formId !== undefined
        ? { formId }
        : {}),

      depth,
    };

    /* ------------------------------------------------------------ *
     * Standard DOM metadata
     * ------------------------------------------------------------ */

    const role =
      el.getAttribute('role');

    if (role) {
      node.role =
        role;
    }

    const type =
      el.getAttribute('type');

    if (type) {
      node.type =
        type;
    }

    const name =
      el.getAttribute('name');

    if (name) {
      node.name =
        name;
    }

    const label =
      labelFor(el);

    if (label) {
      node.label =
        label;
    }

    const placeholder =
      el.getAttribute(
        'placeholder',
      );

    if (placeholder) {
      node.placeholder =
        placeholder;
    }

    if (rect) {
      node.box =
        rect;
    }

    /* ------------------------------------------------------------ *
     * Checked state
     *
     * Do NOT use view.HTMLInputElement here.
     * That was the source of the TypeScript constructor errors.
     * ------------------------------------------------------------ */

    if (
      el.tagName ===
      'INPUT'
    ) {
      const input =
        el as HTMLInputElement;

      if (
        input.type ===
          'checkbox' ||
        input.type ===
          'radio'
      ) {
        node.checked =
          input.checked;
      }
    }

    /* ------------------------------------------------------------ *
     * Disabled state
     * ------------------------------------------------------------ */

    if (
      el.hasAttribute(
        'disabled',
      )
    ) {
      node.disabled =
        true;
    }

    /* ------------------------------------------------------------ *
     * Safe href
     * ------------------------------------------------------------ */

    const href =
      el.getAttribute(
        'href',
      );

    if (href) {
      node.href =
        safeHref(
          href,
          doc,
        );
    }

    /* ------------------------------------------------------------ *
     * Text scrubbing
     * ------------------------------------------------------------ */

    const ownText =
      squash(
        directText(el),
      );

    if (ownText) {
      const {
        text,
        reasons:
          textReasons,
      } = redactText(
        ownText,
        REDACTED_PLACEHOLDER,
      );

      node.text =
        text;

      if (
        textReasons.length >
        0
      ) {
        textReasons.forEach(
          bump,
        );

        node.redacted = [
          ...(node.redacted ??
            []),
          ...textReasons,
        ];

        if (rect) {
          sensitiveBoxes.push(
            rect,
          );
        }
      }
    }

    const rawContext = findContext(el);
    if (rawContext) {
      const { text: cleanContext, reasons: ctxReasons } = redactText(rawContext, REDACTED_PLACEHOLDER);
      if (cleanContext && cleanContext !== ownText && cleanContext !== node.label) {
        node.context = cleanContext.slice(0, 120);
        if (ctxReasons.length > 0) {
          ctxReasons.forEach(bump);
          node.redacted = [...new Set([...(node.redacted ?? []), ...ctxReasons])];
        }
      }
    }

    /* ------------------------------------------------------------ *
     * Form/control value scrubbing
     *
     * Use tag-name checks + casts instead of Window constructors.
     * ------------------------------------------------------------ */

    const rawValue = readValue(el);

    if (rawValue !== undefined) {
      if (isValueForbidden(reasons)) {
        node.value = REDACTED_PLACEHOLDER;
      } else if (hasPii(rawValue)) {
        const found = detectPii(rawValue).map((match) => match.reason);
        found.forEach(bump);
        node.value = REDACTED_PLACEHOLDER;
        node.redacted = [
          ...new Set([
            ...(node.redacted ?? []),
            ...found,
          ]),
        ];
        if (rect) {
          sensitiveBoxes.push(rect);
        }
      } else {
        node.value = squash(rawValue);
      }
    }

    /* ------------------------------------------------------------ *
     * Element-level redaction reasons
     * ------------------------------------------------------------ */

    if (
      reasons.length > 0
    ) {
      node.redacted = [
        ...new Set([
          ...(node.redacted ??
            []),
          ...reasons,
        ]),
      ];
    }

    nodes.push(node);
  }

  /* -------------------------------------------------------------- *
   * Build child relationships
   * -------------------------------------------------------------- */

  const nodeById =
    new Map<number, ScrubbedNode>();

  for (const node of nodes) {
    node.childIds = [];

    nodeById.set(
      node.id,
      node,
    );
  }

  for (const node of nodes) {
    if (
      node.parentId ===
      undefined
    ) {
      continue;
    }

    const parent =
      nodeById.get(
        node.parentId,
      );

    if (!parent) {
      /*
       * Defensive cleanup.
       *
       * A dangling parent relationship must never reach the
       * reasoning layer.
       */
      delete node.parentId;
      node.depth = 0;
      continue;
    }

    parent.childIds!.push(
      node.id,
    );
  }

  /* -------------------------------------------------------------- *
   * Page metadata
   * -------------------------------------------------------------- */

  const location =
    doc.location ??
    ({
      href: '',
      origin: '',
    } as Location);

  const dom: ScrubbedDom = {
    url:
      stripUrl(
        location.href,
      ),

    origin:
      location.origin,

    title:
      squash(
        doc.title ?? '',
      ),

    viewport: {
      width:
        view?.innerWidth ?? 0,

      height:
        view?.innerHeight ?? 0,

      scrollX:
        Math.round(
          view?.scrollX ?? 0,
        ),

      scrollY:
        Math.round(
          view?.scrollY ?? 0,
        ),
    },

    nodes,

    redactionSummary:
      summary,
  };

  return {
    dom,
    sensitiveBoxes,
    index,
  };
}

const CONTAINER_SELECTOR =
  'article, [role="listitem"], [role="article"], li, tr, [data-testid*="product"], [data-testid*="item"], [data-testid*="card"], [class*="product"], [class*="card"], [class*="item"], [class*="result"], [class*="tile"], [class*="listing"], [class*="row"], fieldset, form, section';

/**
 * Finds semantic container context (e.g. product title, card header, table row context)
 * so that identical action buttons like "Add to Cart" or "Select" can be disambiguated.
 */
function findContext(el: Element): string | undefined {
  let container: Element | null = null;
  let curr = el.parentElement;
  let depth = 0;
  while (curr && depth < 5) {
    if (curr.matches && curr.matches(CONTAINER_SELECTOR)) {
      container = curr;
      break;
    }
    curr = curr.parentElement;
    depth++;
  }

  if (!container && el.parentElement) {
    const parent = el.parentElement;
    if (parent.querySelector?.('h1, h2, h3, h4, h5, h6, [class*="title"], [class*="name"], [class*="header"]')) {
      container = parent;
    } else if (parent.parentElement?.querySelector?.('h1, h2, h3, h4, h5, h6, [class*="title"], [class*="name"]')) {
      container = parent.parentElement;
    }
  }

  if (!container) return undefined;

  // 1. Try finding explicit title / heading elements in container
  const titleEl = container.querySelector?.(
    'h1, h2, h3, h4, h5, h6, [class*="title"], [class*="name"], [class*="heading"], [class*="header"], [data-testid*="title"], [data-testid*="name"]'
  );
  if (titleEl && titleEl !== el && !titleEl.contains(el)) {
    const text = squash(titleEl.textContent ?? '');
    if (text && text.length > 2) return text;
  }

  // 2. In table rows, look at first cell (th or td)
  if (container.tagName === 'TR') {
    const firstCell = container.querySelector?.('th, td');
    if (firstCell && firstCell !== el && !firstCell.contains(el)) {
      const text = squash(firstCell.textContent ?? '');
      if (text && text.length > 2) return text;
    }
  }

  // 3. Container aria-label or title attribute
  const aria = container.getAttribute?.('aria-label') || container.getAttribute?.('title');
  if (aria && aria.trim().length > 2) return squash(aria);

  // 4. Extract non-interactive text summary inside the container
  let summary = '';
  for (const child of container.childNodes) {
    if (child.nodeType === 3) summary += (child.nodeValue ?? '') + ' ';
    else if (child.nodeType === 1 && child !== el && !(child as Element).contains(el) && !(child as Element).matches?.(INTERACTIVE)) {
      summary += ((child as Element).textContent ?? '') + ' ';
    }
    if (summary.length > 120) break;
  }
  const cleanSummary = squash(summary);
  if (cleanSummary && cleanSummary.length > 3) return cleanSummary;

  return undefined;
}

/* ------------------------------------------------------------------ *
 * Text extraction
 * ------------------------------------------------------------------ */

function directText(
  el: Element,
): string {
  let out = '';

  for (
    const child of el.childNodes
  ) {
    if (
      child.nodeType === 3
    ) {
      out +=
        child.nodeValue ?? '';
    } else if (
      child.nodeType === 1 &&
      !(
        child as Element
      ).matches(
        INTERACTIVE,
      )
    ) {
      out +=
        (
          child as Element
        ).textContent ?? '';
    }

    if (
      out.length > 400
    ) {
      break;
    }
  }

  return out;
}

/* ------------------------------------------------------------------ *
 * Value extraction
 * ------------------------------------------------------------------ */

function readValue(
  el: Element,
): string | undefined {
  const tag =
    el.tagName.toLowerCase();

  if (tag === 'input') {
    const input =
      el as HTMLInputElement;

    if (
      input.type ===
        'checkbox' ||
      input.type ===
        'radio'
    ) {
      return undefined;
    }

    return input.value;
  }

  if (tag === 'textarea') {
    const textarea =
      el as HTMLTextAreaElement;

    return textarea.value;
  }

  if (tag === 'select') {
    const select =
      el as HTMLSelectElement;

    return select.value;
  }

  if (
    el.getAttribute(
      'contenteditable',
    ) === 'true'
  ) {
    return (
      el.textContent ?? ''
    );
  }

  return undefined;
}

/* ------------------------------------------------------------------ *
 * URL sanitisation
 * ------------------------------------------------------------------ */

export function stripUrl(
  href: string,
): string {
  try {
    const url =
      new URL(href);

    return `${url.origin}${url.pathname}`;
  } catch {
    return '';
  }
}

function safeHref(
  href: string,
  doc: Document,
): string {
  try {
    return stripUrl(
      new URL(
        href,
        doc.location?.href ??
          'https://localhost',
      ).href,
    );
  } catch {
    return '';
  }
}