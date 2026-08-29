/**
 * Semantic Page Model
 *
 * Converts an already-scrubbed DOM snapshot into a stable, human-readable
 * representation that the AI reasoning layer uses.
 *
 * Design principles:
 *  - Reads only already-scrubbed `ScrubbedNode` fields — no raw DOM access.
 *  - No network I/O, no PII. The privacy pipeline runs upstream.
 *  - The `elementId` on each `SemanticElement` equals `ScrubbedNode.id` and
 *    is stable only for the CURRENT page observation. After any navigation or
 *    meaningful DOM mutation the agent must request a fresh observation.
 *  - The format produced by `toPromptText()` is the canonical element
 *    representation used by both the local-VLM prompt and the server prompt.
 */
import type { ScrubbedDom, ScrubbedNode } from '@shared/types';

/* ------------------------------------------------------------------ *
 * Public types
 * ------------------------------------------------------------------ */

/**
 * A single interactive or meaningful element on the current page,
 * expressed in semantic terms the LLM can reason about without knowing
 * anything about the underlying website.
 */
export interface SemanticElement {
  /**
   * Stable numeric ID for this page observation.
   * Equals `ScrubbedNode.id`. Valid only until the next page scrape.
   */
  elementId: number;
  tag: string;
  role?: string;
  type?: string;
  label?: string;
  text?: string;
  placeholder?: string;
  name?: string;
  /** Current control value, already scrubbed. */
  value?: string;
  /** Nearby container text for disambiguation (e.g. product title). */
  context?: string;
  visible: boolean;
  enabled: boolean;
}

export interface PageModel {
  /** All semantic elements for the current observation, in DOM order. */
  elements: SemanticElement[];
  /**
   * Look up a node by its numeric ID.
   * Returns undefined if the ID is not in the current observation.
   */
  byId(id: number): ScrubbedNode | undefined;
  /**
   * Compact, token-efficient text summary for inclusion in LLM prompts.
   * `budget` caps the number of elements emitted (default: all).
   */
  toPromptText(budget?: number): string;
}

/* ------------------------------------------------------------------ *
 * Builder
 * ------------------------------------------------------------------ */

/**
 * Build a semantic page model from an already-scrubbed DOM snapshot.
 *
 * This is a pure function — safe to call from service workers and
 * web workers alike.
 */
export function buildPageModel(dom: ScrubbedDom): PageModel {
  const nodeMap = new Map<number, ScrubbedNode>();
  const elements: SemanticElement[] = [];

  for (const n of dom.nodes) {
    nodeMap.set(n.id, n);
    if (!n.visible) continue;

    elements.push(toSemantic(n));
  }

  return {
    elements,

    byId(id: number): ScrubbedNode | undefined {
      return nodeMap.get(id);
    },

    toPromptText(budget?: number): string {
      const list = budget !== undefined ? elements.slice(0, budget) : elements;
      return list.map(renderElement).join('\n');
    },
  };
}

/* ------------------------------------------------------------------ *
 * Helpers
 * ------------------------------------------------------------------ */

function toSemantic(n: ScrubbedNode): SemanticElement {
  return {
    elementId: n.id,
    tag: n.tag,
    ...(n.role ? { role: n.role } : {}),
    ...(n.type ? { type: n.type } : {}),
    ...(n.label ? { label: n.label } : {}),
    ...(n.text ? { text: n.text } : {}),
    ...(n.placeholder ? { placeholder: n.placeholder } : {}),
    ...(n.name ? { name: n.name } : {}),
    ...(n.value ? { value: n.value } : {}),
    ...(n.context ? { context: n.context } : {}),
    visible: n.visible,
    enabled: !n.disabled,
  };
}

/**
 * One-line element description used inside LLM prompts.
 *
 * Format:
 *   <id>: <tag> [role=<role>] [type=<type>] [label="<label>"] [text="<text>"]
 *          [placeholder="<ph>"] [value="<v>"] [context="<ctx>"] [disabled]
 *
 * This is intentionally verbose enough for the model to reason about the
 * element without seeing any CSS selectors.
 */
function renderElement(el: SemanticElement): string {
  const parts: string[] = [el.tag];
  if (el.role) parts.push(`role=${el.role}`);
  if (el.type && el.type !== el.tag) parts.push(`type=${el.type}`);
  if (el.label) parts.push(`label="${trunc(el.label)}"`);
  if (el.text) parts.push(`text="${trunc(el.text)}"`);
  if (el.placeholder) parts.push(`placeholder="${trunc(el.placeholder)}"`);
  if (el.name) parts.push(`name="${trunc(el.name)}"`);
  if (el.value) parts.push(`value="${trunc(el.value)}"`);
  if (el.context) parts.push(`context="${trunc(el.context, 60)}"`);
  if (!el.enabled) parts.push('disabled');
  return `${el.elementId}: ${parts.join(' ')}`;
}

const trunc = (s: string, n = 48): string =>
  s.length > n ? `${s.slice(0, n)}\u2026` : s;
