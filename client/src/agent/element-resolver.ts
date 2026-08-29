/**
 * Generic Element Resolver
 *
 * Resolves a semantic element target (as decided by the LLM) to a live
 * HTMLElement on the current page.
 *
 * Priority order (per spec §9):
 *   1. elementId  → ScrubbedNode by ID → cached index entry → querySelector
 *   2. ARIA role + accessible name (label)
 *   3. Visible label text
 *   4. Placeholder
 *   5. name attribute + type
 *   6. Nearby semantic context (not implemented — left for future)
 *   7. _legacySelector → direct querySelector (backward compatibility)
 *   8. Vision fallback — not implemented here
 *
 * No website-specific knowledge. No hard-coded selectors.
 * No PII handling — that stays in the privacy pipeline.
 */
import type { ScrubbedDom } from '@shared/types';

/* ------------------------------------------------------------------ *
 * Public surface
 * ------------------------------------------------------------------ */

export interface ResolveTarget {
  /** Primary: ScrubbedNode.id from the current page observation. */
  elementId?: number;
  /** Fallback 1: ARIA role */
  role?: string;
  /** Fallback 2: accessible label (aria-label or associated <label> text) */
  label?: string;
  /** Fallback 3: visible text content of the element */
  text?: string;
  /** Fallback 4: placeholder text */
  placeholder?: string;
  /** Fallback 5: name attribute */
  name?: string;
  /** Fallback 6: input type */
  type?: string;
  /**
   * Fallback 7: legacy CSS selector emitted by backward-compatible paths.
   * Used only when all semantic lookups fail.
   */
  _legacySelector?: string;
}

export class ElementNotFoundError extends Error {
  constructor(target: ResolveTarget) {
    super(
      `[RESOLVER] Could not resolve element: ${JSON.stringify({
        elementId: target.elementId,
        role: target.role,
        label: target.label,
        text: target.text,
        placeholder: target.placeholder,
        name: target.name,
        type: target.type,
        legacy: target._legacySelector,
      })}`,
    );
    this.name = 'ElementNotFoundError';
  }
}

/**
 * Resolve a semantic target to a live HTMLElement.
 *
 * @param target - Semantic description of the element to find.
 * @param nodeIndex - The selector→Element map produced by `buildScrubbedDom`.
 *                   Lives only in the content script; never crosses message channels.
 * @param dom - The current `ScrubbedDom` snapshot, used for ID lookup.
 * @throws {ElementNotFoundError} when no element can be found.
 */
export function resolveElement(
  target: ResolveTarget,
  nodeIndex: Map<string, Element>,
  dom: ScrubbedDom,
): HTMLElement {
  // --- Path 1: elementId ---------------------------------------------------
  if (target.elementId !== undefined) {
    const node = dom.nodes.find((n) => n.id === target.elementId);
    if (node) {
      const byIndex = nodeIndex.get(node.selector);
      if (byIndex?.isConnected) {
        console.debug(
          `[RESOLVER] elementId=${target.elementId} → ${node.tag} via index`,
        );
        return operable(byIndex as HTMLElement);
      }
      try {
        const byCss = document.querySelector<HTMLElement>(node.selector);
        if (byCss) {
          console.debug(
            `[RESOLVER] elementId=${target.elementId} → ${node.tag} via querySelector`,
          );
          return operable(byCss);
        }
      } catch {
        // invalid selector — fall through
      }
    }
  }

  // --- Path 2: role + label ------------------------------------------------
  if (target.role && target.label) {
    const el = findByRoleAndLabel(target.role, target.label);
    if (el) {
      console.debug(
        `[RESOLVER] role=${target.role} label="${target.label}" → ${el.tagName}`,
      );
      return operable(el);
    }
  }

  // --- Path 3: label only --------------------------------------------------
  if (target.label) {
    const el = findByLabel(target.label);
    if (el) {
      console.debug(`[RESOLVER] label="${target.label}" → ${el.tagName}`);
      return operable(el);
    }
  }

  // --- Path 4: visible text ------------------------------------------------
  if (target.text) {
    const el = findByText(target.text);
    if (el) {
      console.debug(`[RESOLVER] text="${target.text}" → ${el.tagName}`);
      return operable(el);
    }
  }

  // --- Path 5: placeholder -------------------------------------------------
  if (target.placeholder) {
    const el = document.querySelector<HTMLElement>(
      `[placeholder="${CSS.escape(target.placeholder)}"]`,
    );
    if (el) {
      console.debug(
        `[RESOLVER] placeholder="${target.placeholder}" → ${el.tagName}`,
      );
      return operable(el);
    }
  }

  // --- Path 6: name + type -------------------------------------------------
  if (target.name) {
    const sel = target.type
      ? `[name="${CSS.escape(target.name)}"][type="${CSS.escape(target.type)}"]`
      : `[name="${CSS.escape(target.name)}"]`;
    try {
      const el = document.querySelector<HTMLElement>(sel);
      if (el) {
        console.debug(
          `[RESOLVER] name="${target.name}" type="${target.type ?? '*'}" → ${el.tagName}`,
        );
        return operable(el);
      }
    } catch {
      // ignore
    }
  }

  // --- Path 7: legacy CSS selector -----------------------------------------
  if (target._legacySelector) {
    try {
      const cached = nodeIndex.get(target._legacySelector);
      if (cached?.isConnected) {
        console.debug(
          `[RESOLVER] legacy selector "${target._legacySelector}" → via index`,
        );
        return operable(cached as HTMLElement);
      }
      const el = document.querySelector<HTMLElement>(target._legacySelector);
      if (el) {
        console.debug(
          `[RESOLVER] legacy selector "${target._legacySelector}" → querySelector`,
        );
        return operable(el);
      }
    } catch {
      // ignore invalid selector
    }

    // ID fallback: #foo → getElementById('foo')
    if (target._legacySelector.startsWith('#')) {
      const rawId = target._legacySelector
        .slice(1)
        .replace(/\\/g, '')
        .trim();
      const byId = document.getElementById(rawId);
      if (byId) {
        console.debug(
          `[RESOLVER] legacy selector "${target._legacySelector}" → getElementById`,
        );
        return operable(byId);
      }
    }
  }

  throw new ElementNotFoundError(target);
}

/* ------------------------------------------------------------------ *
 * Semantic lookup helpers
 * ------------------------------------------------------------------ */

/** Find the first interactive element with a matching ARIA role + label. */
function findByRoleAndLabel(role: string, label: string): HTMLElement | null {
  const needle = label.toLowerCase().trim();
  const candidates = document.querySelectorAll<HTMLElement>(`[role="${CSS.escape(role)}"]`);
  for (const el of candidates) {
    const ariaLabel = el.getAttribute('aria-label') ?? '';
    const labelledBy = el.getAttribute('aria-labelledby')
      ? (document.getElementById(el.getAttribute('aria-labelledby')!)?.textContent ?? '')
      : '';
    const visible = el.textContent ?? '';
    const hay = `${ariaLabel} ${labelledBy} ${visible}`.toLowerCase();
    if (hay.includes(needle)) return el;
  }
  return null;
}

/** Find the first element with a matching accessible label. */
function findByLabel(label: string): HTMLElement | null {
  const needle = label.toLowerCase().trim();

  // aria-label
  const byAria = document.querySelector<HTMLElement>(
    `[aria-label="${CSS.escape(label)}"]`,
  );
  if (byAria) return byAria;

  // <label for="..."> association
  const labels = document.querySelectorAll<HTMLLabelElement>('label');
  for (const l of labels) {
    if (l.textContent?.toLowerCase().trim().includes(needle)) {
      const target = l.htmlFor
        ? document.getElementById(l.htmlFor)
        : l.querySelector<HTMLElement>('input, select, textarea');
      if (target) return target as HTMLElement;
    }
  }

  return null;
}

/** Find an interactive element by its trimmed visible text. */
function findByText(text: string): HTMLElement | null {
  const needle = text.toLowerCase().trim();
  const INTERACTIVE_SELECTORS =
    'button, a, [role="button"], [role="link"], [role="menuitem"], [role="tab"], summary';
  const candidates = document.querySelectorAll<HTMLElement>(INTERACTIVE_SELECTORS);
  for (const el of candidates) {
    const visible = (el.textContent ?? '').toLowerCase().trim();
    if (visible === needle || visible.startsWith(needle)) return el;
  }
  return null;
}

/* ------------------------------------------------------------------ *
 * Operable element coercion
 * ------------------------------------------------------------------ */

/**
 * If the resolved element is a decorative child (span/icon/svg) inside a
 * button or link, return the operable container instead.
 * Identical logic to the existing `getOperableElement` in content/index.ts.
 */
function operable(el: HTMLElement): HTMLElement {
  if (
    el.tagName === 'BUTTON' ||
    el.tagName === 'A' ||
    el.tagName === 'INPUT' ||
    el.tagName === 'SELECT' ||
    el.tagName === 'TEXTAREA' ||
    el.getAttribute('role') === 'button'
  ) {
    return el;
  }
  const parent = el.closest<HTMLElement>(
    'button, a, [role="button"], input, select, textarea, summary',
  );
  return parent ?? el;
}
