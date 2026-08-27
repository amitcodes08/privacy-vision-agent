/**
 * Cheap, deterministic task-completion checker.
 *
 * Runs on already-available data (ScrubbedDom, current URL/title, last action,
 * action history). Zero extra network I/O, zero extra VLM calls.
 *
 * Design constraints from the architecture:
 *   - Must never send raw pixels anywhere.
 *   - Must never inspect or log password / OTP / credit-card values.
 *   - Must never bypass the existing redaction pipeline.
 *   - Must return "not done" when evidence is ambiguous — false negatives are
 *     far cheaper than false positives.
 *
 * The caller (background/index.ts) uses the signal as follows:
 *   confidence >= HIGH_CONFIDENCE → stop immediately (before VLM call)
 *   VLM returns done + deterministic evidence → strengthen trust
 *   otherwise → let the normal decision path run
 */
import type { AgentAction, ScrubbedDom } from '@shared/types';
import { literalFor, tokenize } from '~/ai/local-planner';

/* ------------------------------------------------------------------ *
 * Public surface
 * ------------------------------------------------------------------ */

export interface TerminationSignal {
  /** Whether the checker believes the goal is satisfied. */
  done: boolean;
  /**
   * 0..1 — checker confidence in its own answer.
   * NOTE: this is NOT the VLM action confidence; do not mix the two.
   * Only stop early when this exceeds HIGH_CONFIDENCE (0.75).
   */
  confidence: number;
  /** Human-readable reason, safe to log (no raw values). */
  reason: string;
}

/**
 * The minimum checker confidence required for the agent to stop
 * *before* invoking the VLM. Below this threshold the signal is used
 * only to corroborate a VLM `done`.
 */
export const HIGH_CONFIDENCE = 0.75;

export interface CheckInput {
  goal: string;
  dom: ScrubbedDom;
  /** The action that produced this DOM (the one we just executed). */
  lastAction?: AgentAction;
  history?: AgentAction[];
  /** DOM from before the last action, when available, for delta checks. */
  prevDom?: ScrubbedDom;
  taskMemory?: import('@shared/types').TaskMemory;
}

/**
 * Entry point. Runs all sub-checks and returns the highest-confidence signal
 * (or the first one whose `done` is true and confidence is actionable).
 *
 * Never throws — a buggy check must not abort the agent run.
 */
export function checkTermination(input: CheckInput): TerminationSignal {
  try {
    return _check(input);
  } catch {
    return UNKNOWN;
  }
}

/* ------------------------------------------------------------------ *
 * Constants
 * ------------------------------------------------------------------ */

const UNKNOWN: TerminationSignal = {
  done: false,
  confidence: 0,
  reason: 'goal not deterministically verifiable',
};

const NOT_DONE = (reason: string): TerminationSignal => ({ done: false, confidence: 0, reason });
const DONE = (confidence: number, reason: string): TerminationSignal => ({ done: true, confidence, reason });

const COMPOUND_GOAL_PATTERN = /\b(?:and\s+then|then|after\s+that|followed\s+by|;\s*|\s*->\s*)\b/i;
const MULTI_ACTION_PATTERN = /\b(?:search|find|navigate|go\s+to|visit)\b.*\b(?:and\s+|,\s*and\s+)(?:add|checkout|click|fill|type|buy|select|submit|choose)\b/i;

function isCompoundGoal(goal: string, taskMemory?: import('@shared/types').TaskMemory): boolean {
  if (taskMemory?.subObjectives && taskMemory.subObjectives.length > 1) {
    return true;
  }
  return COMPOUND_GOAL_PATTERN.test(goal) || MULTI_ACTION_PATTERN.test(goal);
}

/* ------------------------------------------------------------------ *
 * Orchestrator
 * ------------------------------------------------------------------ */

function _check(input: CheckInput): TerminationSignal {
  const { goal, dom, lastAction, prevDom, history = [], taskMemory } = input;

  if (lastAction?.action === 'done') return DONE(1, 'last action was done');
  if (lastAction?.action === 'wait') return NOT_DONE('last action was wait — page may still be loading');

  // Multi-step safety: if there are pending sub-objectives, the overarching task cannot be complete
  if (taskMemory?.subObjectives && taskMemory.subObjectives.length > 1) {
    const pending = taskMemory.subObjectives.filter((s) => s.status !== 'completed');
    if (pending.length > 0) {
      return NOT_DONE(`${pending.length} sub-objectives still pending`);
    }
  }

  const compound = isCompoundGoal(goal, taskMemory);
  const literal = literalFor(goal);
  const tokens = tokenize(goal);

  // 1. Literal Evidence (e.g., "Add item 'Buy groceries'")
  if (literal) {
    const literalLower = literal.toLowerCase();
    
    let foundInStaticState = false;
    let foundInInput = false;

    // Check title and URL first only for non-compound goals
    if (!compound && (dom.title.toLowerCase().includes(literalLower) || 
        decodeURIComponent(dom.url).toLowerCase().includes(literalLower))) {
      foundInStaticState = true;
    }

    for (const n of dom.nodes) {
      const hay = [n.label, n.text, n.name].filter(Boolean).join(' ').toLowerCase();
      if (hay.includes(literalLower) || (n.value && n.value.toLowerCase().includes(literalLower))) {
        // If it's an editable field, it might just be sitting there unsubmitted
        if (n.tag === 'input' || n.tag === 'textarea' || n.role === 'textbox' || n.role === 'combobox' || n.role === 'searchbox') {
           foundInInput = true;
        } else {
           foundInStaticState = true;
        }
      }
    }

    // If the literal appears in the static DOM (e.g. list item, heading), that is strong positive evidence.
    // For compound goals (e.g. search + add to cart), finding the search term is not sufficient to declare the whole task done.
    if (foundInStaticState && !compound) {
      return DONE(0.8, `found requested literal "${literal}" in page state`);
    }

    // If it's completely missing (not even in an input) and we've already taken actions, that's strong negative evidence.
    if (!foundInStaticState && !foundInInput && history.length > 0) {
      return {
        done: false,
        confidence: 0.8, // Overrides VLM
        reason: `requested literal "${literal}" is completely missing from the page`
      };
    }
  }

  // 2. State Change Evidence (e.g., "Navigate to billing")
  // Only apply navigation token matching for single-step navigation goals, not compound workflows
  if (!compound && history.length > 0 && prevDom && tokens.length > 0) {
    const prevUrl = new URL(prevDom.url);
    const currUrl = new URL(dom.url);
    
    // Did we navigate or did the page title change?
    if (prevUrl.pathname !== currUrl.pathname || prevDom.title !== dom.title) {
      const pageText = `${dom.title} ${currUrl.pathname}`.toLowerCase();
      const matchCount = tokens.filter(t => pageText.includes(t)).length;
      
      // If the new state contains a good chunk of the goal tokens, consider it done.
      if (matchCount > 0 && matchCount >= tokens.length / 2) {
         return DONE(0.8, 'page navigated/changed to a state matching goal tokens');
      }
    }
  }

  return UNKNOWN;
}

/* ------------------------------------------------------------------ *
 * VLM / DOM corroboration
 * ------------------------------------------------------------------ */

/**
 * Cross-check a VLM `done` signal against DOM evidence.
 *
 * Rules:
 *   - VLM done + DOM done (HIGH_CONFIDENCE) → strong combined done
 *   - VLM done + DOM not-done (high confidence NOT-done) → contradiction;
 *     return the NOT-done signal so the agent continues
 *   - VLM done + DOM unknown/uncertain → return VLM done as-is (caller decides)
 *
 * Calling this is cheap: it reuses an already-computed `domSignal`.
 * Does NOT invoke another VLM inference or network request.
 */
export function corroborateDone(
  vlmConfidence: number,
  domSignal: TerminationSignal,
): TerminationSignal {
  if (domSignal.done && domSignal.confidence >= HIGH_CONFIDENCE) {
    // Both agree: raise confidence to make the caller stop.
    return DONE(
      Math.min(1, (vlmConfidence + domSignal.confidence) / 2 + 0.15),
      `vlm+dom both done: ${domSignal.reason}`,
    );
  }

  if (!domSignal.done && domSignal.confidence >= HIGH_CONFIDENCE) {
    // Contradiction: DOM is highly confident the goal is NOT satisfied.
    // Trust the DOM over the VLM — it has direct structural evidence.
    return {
      done: false,
      confidence: domSignal.confidence,
      reason: `vlm says done but dom contradicts: ${domSignal.reason}`,
    };
  }

  // DOM is uncertain.
  // Do NOT blindly trust a low-confidence VLM done.
  // We require HIGH_CONFIDENCE from the VLM to accept a done signal
  // that lacks independent DOM verification.
  if (vlmConfidence < HIGH_CONFIDENCE) {
    return {
      done: false,
      confidence: 0.6,
      reason: 'vlm done lacks high confidence, and dom evidence is insufficient to verify',
    };
  }

  // VLM is highly confident, DOM is uncertain. Trust VLM.
  return {
    done: true,
    confidence: vlmConfidence,
    reason: 'vlm done (dom evidence insufficient to confirm or contradict)',
  };
}

/* ------------------------------------------------------------------ *
 * Helpers
 * ------------------------------------------------------------------ */


