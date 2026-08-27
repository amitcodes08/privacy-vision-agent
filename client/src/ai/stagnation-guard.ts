/**
 * Lightweight page-stagnation guard.
 *
 * The existing `repeatedTail` check catches the same action back-to-back, but
 * misses patterns like:  click A → scroll → click B → scroll → click C
 * where every action is distinct yet the page never meaningfully changes.
 *
 * This module adds a structural DOM fingerprint. Two identical fingerprints
 * separated by ≥ STAGNATION_WINDOW distinct non-wait actions is stagnation.
 *
 * Privacy constraints:
 *   - The fingerprint is derived from structural counts and the stripped URL /
 *     title — never from raw field values, redacted text, or PII.
 *   - Safe to log (no personal data, no raw DOM content).
 */
import type { AgentAction, ScrubbedDom } from '@shared/types';

/* ------------------------------------------------------------------ *
 * Configuration
 * ------------------------------------------------------------------ */

/**
 * How many distinct (non-wait) actions must occur between two identical
 * fingerprints before we call it stagnation.
 * Set high enough that legitimate multi-step tasks (fill → scroll → click)
 * are not falsely flagged, yet low enough to catch real loops.
 */
const STAGNATION_WINDOW = 3;

/**
 * How many steps of grace to give after a `navigate` action before the
 * fingerprint comparison resumes. During a page load the DOM is in flux.
 */
const POST_NAVIGATE_GRACE = 2;

/* ------------------------------------------------------------------ *
 * Public surface
 * ------------------------------------------------------------------ */

export interface StagnationState {
  /** Structural fingerprints, one per step, in order. */
  fingerprints: string[];
  /** Non-wait action count, parallel to fingerprints. */
  actionCounts: number[];
  /** Total non-wait actions seen so far. */
  totalNonWait: number;
  /** Step index of the last navigate action (or -1). */
  lastNavigateStep: number;
}

export function makeStagnationState(): StagnationState {
  return { fingerprints: [], actionCounts: [], totalNonWait: 0, lastNavigateStep: -1 };
}

/**
 * Record a new observation and check for stagnation.
 *
 * @returns true when the page appears to be stuck.
 */
export function recordAndCheck(
  state: StagnationState,
  dom: ScrubbedDom,
  action: AgentAction,
  stepIndex: number,
): boolean {
  const isWait = action.action === 'wait';

  if (action.action === 'navigate') {
    state.lastNavigateStep = stepIndex;
  }

  if (!isWait) {
    // Only record fingerprints for non-wait actions. Wait steps are intentional
    // pauses for async page loading and should not contribute to the stagnation
    // streak — a DOM that looks the same during a wait is expected.
    state.totalNonWait++;
    const fp = fingerprint(dom);
    state.fingerprints.push(fp);
    state.actionCounts.push(state.totalNonWait);
  }

  return isStagnating(state, stepIndex);
}

/* ------------------------------------------------------------------ *
 * Fingerprint
 * ------------------------------------------------------------------ */

/**
 * Structural fingerprint of a ScrubbedDom. Intentionally coarse so that minor
 * DOM mutations (e.g. a tooltip appearing) do not break the equality check.
 *
 * Components:
 *   - stripped URL (already token-free from dom-scrubber)
 *   - page title (first 40 chars)
 *   - scroll position bucket (every 200px)
 *   - visible node count by tag (not raw text, not values)
 *   - redaction summary counts (available, non-sensitive by design)
 */
export function fingerprint(dom: ScrubbedDom): string {
  const tagCounts = new Map<string, number>();
  for (const n of dom.nodes) {
    if (n.visible) tagCounts.set(n.tag, (tagCounts.get(n.tag) ?? 0) + 1);
  }

  const tagPart = [...tagCounts.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([tag, count]) => `${tag}:${count}`)
    .join(',');

  const scrollBucket = Math.floor(dom.viewport.scrollY / 200);

  return [
    dom.url,
    dom.title.slice(0, 40),
    `scroll:${scrollBucket}`,
    tagPart,
  ].join('|');
}

/* ------------------------------------------------------------------ *
 * Stagnation detection
 * ------------------------------------------------------------------ */

function isStagnating(state: StagnationState, currentStep: number): boolean {
  const { fingerprints, lastNavigateStep } = state;
  const len = fingerprints.length;
  // Need at least STAGNATION_WINDOW + 1 entries to form a streak.
  if (len < STAGNATION_WINDOW + 1) return false;

  // Don't compare while still in the post-navigate grace window.
  const gracedUntil = lastNavigateStep >= 0 ? lastNavigateStep + POST_NAVIGATE_GRACE : -1;
  if (lastNavigateStep >= 0 && currentStep <= gracedUntil) return false;

  // Stagnation = the last (STAGNATION_WINDOW + 1) contiguous fingerprints are
  // all identical. Using a contiguous window means a DOM change anywhere in the
  // window resets the streak — so a page that changes and then returns to a
  // previous state does NOT trigger stagnation until STAGNATION_WINDOW more
  // identical-fingerprint steps happen after it settles.
  const current = fingerprints[len - 1]!;
  for (let i = len - 2; i >= len - 1 - STAGNATION_WINDOW; i--) {
    if (i < 0) return false;
    if (fingerprints[i] !== current) return false;
  }
  return true;
}
