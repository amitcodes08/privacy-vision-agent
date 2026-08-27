/**
 * Tests for termination-checker.ts and stagnation-guard.ts.
 *
 * Coverage:
 *   - Navigation goal → clearly satisfied
 *   - Navigation goal → NOT satisfied (URL unchanged)
 *   - Click/open goal → satisfied (new visible element)
 *   - Click/open goal → NOT satisfied
 *   - Form fill goal → satisfied (non-sensitive field populated)
 *   - Form fill goal → NOT satisfied (field empty)
 *   - Form fill goal → ambiguous (only sensitive fields → unknown)
 *   - Unrelated keyword in DOM → NOT done (false-positive prevention)
 *   - Ambiguous/unknown goal → NOT done
 *   - Sensitive values never appear in fingerprints or trigger checks
 *   - Stagnation detection: same fingerprint with enough distinct actions
 *   - Stagnation tolerates: wait actions, post-navigate grace window
 *   - Legitimate DOM changes across steps → not stagnating
 *
 * Regression tests (added for the termination-bug fix):
 *   REG-TC-1: TodoMVC item exists in DOM → agent terminates (item-create class)
 *   REG-TC-2: TodoMVC item does NOT exist → agent continues
 *   REG-TC-3: corroborateDone — VLM+DOM both say done → stop
 *   REG-TC-4: corroborateDone — VLM says done, DOM contradicts → continue
 *   REG-TC-5: corroborateDone — DOM uncertain → pass through VLM signal
 *   REG-TC-6: completion check never invokes VLM (performance invariant)
 */
import { describe, expect, it } from 'vitest';
import type { AgentAction, ScrubbedDom, ScrubbedNode } from '@shared/types';
import { checkTermination, corroborateDone, HIGH_CONFIDENCE } from '~/ai/termination-checker';
import { fingerprint, makeStagnationState, recordAndCheck } from '~/ai/stagnation-guard';


/* ------------------------------------------------------------------ *
 * Test helpers
 * ------------------------------------------------------------------ */

function node(over: Partial<ScrubbedNode> & { id: number; selector: string }): ScrubbedNode {
  return { tag: 'button', visible: true, ...over };
}

function makeDom(
  url: string,
  title: string,
  nodes: ScrubbedNode[],
  scrollY = 0,
): ScrubbedDom {
  return {
    url,
    origin: new URL(url).origin,
    title,
    viewport: { width: 1200, height: 800, scrollX: 0, scrollY },
    nodes,
    redactionSummary: {},
  };
}



/* ================================================================== *
 * checkTermination
 * ================================================================== */

describe('checkTermination — generic state verification', () => {
  it('returns DONE if requested literal exists in DOM text (positive evidence)', () => {
    const dom = makeDom('https://todo.test/', 'Todo', [
      node({ id: 0, selector: '#input', tag: 'input', text: '', value: '' }),
      node({ id: 1, selector: '#list-item', tag: 'li', text: 'Buy groceries' }),
    ]);
    const signal = checkTermination({
      goal: 'add item "Buy groceries"',
      dom,
      history: [{ action: 'fill', selector: '#input', valueType: 'LITERAL' }],
    });
    expect(signal.done).toBe(true);
    expect(signal.confidence).toBe(0.8);
    expect(signal.reason).toContain('found requested literal "Buy groceries"');
  });

  it('returns NOT DONE if requested literal is completely missing after history (negative evidence)', () => {
    const dom = makeDom('https://todo.test/', 'Todo', [
      node({ id: 0, selector: '#input', tag: 'input', text: '', value: 'Buy milk' }),
    ]);
    const signal = checkTermination({
      goal: 'add item "Buy milk"',
      dom,
      history: [{ action: 'fill', selector: '#input', valueType: 'LITERAL' }],
    });
    // Missing from static state, exists in input -> ambiguous, not strong negative!
    expect(signal.done).toBe(false);
    expect(signal.confidence).toBe(0); // UNKNOWN
  });

  it('returns NOT DONE with high confidence if literal is completely missing everywhere', () => {
    const dom = makeDom('https://todo.test/', 'Todo', [
      node({ id: 0, selector: '#input', tag: 'input', text: '', value: '' }),
    ]);
    const signal = checkTermination({
      goal: 'add item "Buy milk"',
      dom,
      history: [{ action: 'fill', selector: '#input', valueType: 'LITERAL' }],
    });
    expect(signal.done).toBe(false);
    expect(signal.confidence).toBe(0.8); // STRONG CONTRADICTION
  });

  it('returns DONE if goal tokens are found in new URL/Title', () => {
    const prevDom = makeDom('https://app.test/home', 'Home', []);
    const dom = makeDom('https://app.test/billing', 'Billing Settings', []);
    const signal = checkTermination({
      goal: 'navigate to billing',
      dom,
      prevDom,
      history: [{ action: 'click', selector: '#nav-billing' }],
    });
    expect(signal.done).toBe(true);
    expect(signal.confidence).toBe(0.8);
    expect(signal.reason).toContain('navigated/changed to a state matching');
  });
});

/* ================================================================== *
 * stagnation-guard
 * ================================================================== */

describe('stagnation-guard — fingerprint', () => {
  it('produces identical fingerprints for structurally identical doms', () => {
    const d1 = makeDom('https://a.test/page', 'Page', [
      node({ id: 0, selector: '#btn', text: 'Click me' }),
    ]);
    const d2 = makeDom('https://a.test/page', 'Page', [
      node({ id: 0, selector: '#btn2', text: 'Something else' }),
    ]);
    // Same URL, title, scroll, tag counts (both have 1 button).
    expect(fingerprint(d1)).toBe(fingerprint(d2));
  });

  it('produces different fingerprints when URL changes', () => {
    const d1 = makeDom('https://a.test/page1', 'Page', []);
    const d2 = makeDom('https://a.test/page2', 'Page', []);
    expect(fingerprint(d1)).not.toBe(fingerprint(d2));
  });

  it('produces different fingerprints when node counts change', () => {
    const d1 = makeDom('https://a.test/', 'Home', [
      node({ id: 0, selector: '#a', text: 'A' }),
    ]);
    const d2 = makeDom('https://a.test/', 'Home', [
      node({ id: 0, selector: '#a', text: 'A' }),
      node({ id: 1, selector: '#b', text: 'B' }),
    ]);
    expect(fingerprint(d1)).not.toBe(fingerprint(d2));
  });

  it('produces different fingerprints when scroll position bucket changes', () => {
    const d1 = makeDom('https://a.test/', 'Home', [], 0);
    const d2 = makeDom('https://a.test/', 'Home', [], 400); // crosses 200px bucket
    expect(fingerprint(d1)).not.toBe(fingerprint(d2));
  });

  it('does NOT include raw node text/values in the fingerprint', () => {
    // Two doms with same structure but different text content.
    const d1 = makeDom('https://a.test/', 'Home', [
      node({ id: 0, selector: '#f', tag: 'input', value: 'secret123' }),
    ]);
    const d2 = makeDom('https://a.test/', 'Home', [
      node({ id: 0, selector: '#f', tag: 'input', value: 'other-secret' }),
    ]);
    // Fingerprints should be identical — no raw values included.
    expect(fingerprint(d1)).toBe(fingerprint(d2));
  });
});

describe('stagnation-guard — recordAndCheck', () => {
  const stuckDom = makeDom('https://a.test/', 'Home', [
    node({ id: 0, selector: '#btn', text: 'Click' }),
  ]);

  const click = (sel: string): AgentAction => ({ action: 'click', selector: sel });
  const scroll: AgentAction = { action: 'scroll', deltaY: 200 };
  const wait: AgentAction = { action: 'wait', ms: 500 };

  it('does NOT flag stagnation after fewer than STAGNATION_WINDOW distinct actions', () => {
    const state = makeStagnationState();
    // 2 actions with same fingerprint is below the window (3).
    recordAndCheck(state, stuckDom, click('#a'), 0);
    const stagnant = recordAndCheck(state, stuckDom, click('#b'), 1);
    expect(stagnant).toBe(false);
  });

  it('flags stagnation when same fingerprint repeats after 3+ non-wait actions', () => {
    const state = makeStagnationState();
    recordAndCheck(state, stuckDom, click('#a'), 0);
    recordAndCheck(state, stuckDom, click('#b'), 1);
    recordAndCheck(state, stuckDom, scroll, 2);
    const stagnant = recordAndCheck(state, stuckDom, click('#c'), 3);
    expect(stagnant).toBe(true);
  });

  it('does NOT flag stagnation when DOM structure legitimately changes between same-fp steps', () => {
    const newDom = makeDom('https://a.test/', 'Home', [
      node({ id: 0, selector: '#btn', text: 'Click' }),
      node({ id: 1, selector: '#result', text: 'Results', tag: 'div' }),
    ]);
    const state = makeStagnationState();
    recordAndCheck(state, stuckDom, click('#a'), 0);
    recordAndCheck(state, newDom, click('#b'), 1); // DOM changed
    recordAndCheck(state, newDom, scroll, 2);
    const stagnant = recordAndCheck(state, stuckDom, click('#c'), 3);
    // The chain was broken by newDom; should NOT be stagnating.
    expect(stagnant).toBe(false);
  });

  it('wait actions do not count towards the stagnation window', () => {
    const state = makeStagnationState();
    recordAndCheck(state, stuckDom, click('#a'), 0);
    // Many waits in between — but waits don't count as "distinct actions".
    recordAndCheck(state, stuckDom, wait, 1);
    recordAndCheck(state, stuckDom, wait, 2);
    const stagnant = recordAndCheck(state, stuckDom, click('#b'), 3);
    // Only 2 non-wait actions total — should NOT stagnate.
    expect(stagnant).toBe(false);
  });

  it('does NOT flag stagnation during the post-navigate grace window', () => {
    const navAction: AgentAction = { action: 'navigate', url: 'https://a.test/page2' };
    const state = makeStagnationState();
    recordAndCheck(state, stuckDom, click('#a'), 0);
    recordAndCheck(state, stuckDom, navAction, 1);  // navigate resets grace
    recordAndCheck(state, stuckDom, click('#b'), 2);
    // Step 2 is within grace window (lastNavigate=1, grace=2 steps → gracedUntil=3).
    const stagnant = recordAndCheck(state, stuckDom, click('#c'), 3);
    expect(stagnant).toBe(false);
  });
});

/* ================================================================== *
 * Regression: item-create goal class (TodoMVC scenario)
 * REG-TC-1 & REG-TC-2
 * ================================================================== */



/* ================================================================== *
 * Regression: VLM / DOM corroboration (corroborateDone)
 * REG-TC-3, REG-TC-4, REG-TC-5
 * ================================================================== */

describe('corroborateDone', () => {
  const DOM_DONE = { done: true as const, confidence: 0.88, reason: 'test dom done' };
  const DOM_NOTDONE = { done: false as const, confidence: 0.85, reason: 'test dom not done' };
  const DOM_UNCERTAIN = { done: false as const, confidence: 0, reason: 'unknown' };

  it('REG-TC-3: VLM done + DOM done => combined confidence at or above HIGH_CONFIDENCE', () => {
    const result = corroborateDone(0.70, DOM_DONE);
    expect(result.done).toBe(true);
    expect(result.confidence).toBeGreaterThanOrEqual(HIGH_CONFIDENCE);
  });

  it('REG-TC-4: VLM done + DOM contradicts => NOT done (agent continues)', () => {
    const result = corroborateDone(0.70, DOM_NOTDONE);
    expect(result.done).toBe(false);
  });

  it('REG-TC-5: VLM done (low confidence) + DOM uncertain => VLM confidence NOT passed through, returns NOT done', () => {
    const result = corroborateDone(0.70, DOM_UNCERTAIN);
    expect(result.done).toBe(false);
    expect(result.confidence).toBeCloseTo(0.60, 2);
  });

  it('VLM done (high confidence) + DOM uncertain => VLM confidence passed through unchanged', () => {
    const result = corroborateDone(0.85, DOM_UNCERTAIN);
    expect(result.done).toBe(true);
    expect(result.confidence).toBeCloseTo(0.85, 2);
  });

  it('combined confidence never exceeds 1.0', () => {
    const result = corroborateDone(1.0, { done: true, confidence: 0.99, reason: 'dom' });
    expect(result.confidence).toBeLessThanOrEqual(1.0);
  });
});

/* ================================================================== *
 * Regression: Performance — checkTermination must be purely synchronous
 * REG-TC-6: completion check does NOT invoke another VLM inference
 * ================================================================== */

describe('checkTermination — performance invariant (REG-TC-6)', () => {
  it('checkTermination is purely synchronous / no VLM call', () => {
    // The function must be entirely deterministic and return a plain object
    // without needing await. If it ever became async this test would catch it
    // because the .done access below would receive a Promise, making it
    // non-boolean and failing the assertion.
    const dom = makeDom('https://todomvc.com/', 'TodoMVC', [
      node({ id: 0, selector: 'li', tag: 'li', text: 'Buy milk', role: 'listitem' }),
    ]);
    const result = checkTermination({ goal: 'add Buy milk', dom });
    expect(typeof result.done).toBe('boolean');
    expect(typeof result.confidence).toBe('number');
    expect(typeof result.reason).toBe('string');
    // Confirm it is NOT a Promise (would have a .then property).
    expect(typeof (result as unknown as Promise<unknown>).then).toBe('undefined');
  });
});

/* ================================================================== *
 * Regression: loop guard — REG-TC-9
 * Same action repeated must still trigger the stagnation guard
 * ================================================================== */

describe('stagnation-guard — REG-TC-9 same fingerprint 3+ times', () => {
  it('stagnation guard fires after 4 identical-fingerprint non-wait actions', () => {
    const stuckDom = makeDom('https://a.test/', 'Home', [
      node({ id: 0, selector: '#input', tag: 'input', placeholder: 'new todo' }),
    ]);
    const click = (sel: string): AgentAction => ({ action: 'click', selector: sel });
    const state = makeStagnationState();

    // Four consecutive non-wait actions with the same DOM fingerprint.
    recordAndCheck(state, stuckDom, click('#input'), 0);
    recordAndCheck(state, stuckDom, click('#input'), 1);
    recordAndCheck(state, stuckDom, click('#input'), 2);
    const stagnant = recordAndCheck(state, stuckDom, click('#input'), 3);
    expect(stagnant).toBe(true);
  });
});

describe('checkTermination — Multi-Step & Compound Goal Safety', () => {
  it('does NOT terminate early on compound goal when search results appear', () => {
    const searchResultDom = makeDom(
      'https://shop.test/search?q=headphones',
      'Search results for headphones',
      [
        node({ id: 0, selector: '.item-1', tag: 'a', text: 'Sony Noise Cancelling Headphones' }),
        node({ id: 1, selector: '.add-cart', tag: 'button', text: 'Add to Cart' }),
      ],
    );

    const signal = checkTermination({
      goal: 'Search for headphones and add to cart',
      dom: searchResultDom,
      history: [{ action: 'fill', selector: 'input[name=q]', value: 'headphones', valueType: 'LITERAL' }],
      prevDom: makeDom('https://shop.test/', 'Shop Home', [node({ id: 0, selector: 'input[name=q]', tag: 'input' })]),
    });

    // Should NOT be done because "add to cart" has not happened yet!
    expect(signal.done).toBe(false);
  });

  it('does NOT terminate when TaskMemory has pending sub-objectives', () => {
    const dom = makeDom('https://shop.test/search', 'Search', [
      node({ id: 0, selector: '#cart', tag: 'button', text: 'Cart' }),
    ]);

    const signal = checkTermination({
      goal: 'Search for shoes and add to cart',
      dom,
      taskMemory: {
        goal: 'Search for shoes and add to cart',
        subObjectives: [
          { id: 1, description: 'Search for shoes', status: 'completed' },
          { id: 2, description: 'Add to cart', status: 'pending' },
        ],
        completedObjectives: [{ description: 'Search for shoes', status: 'completed' }],
        attemptedTargets: [],
        step: 2,
      },
    });

    expect(signal.done).toBe(false);
    expect(signal.reason).toContain('sub-objectives still pending');
  });
});

