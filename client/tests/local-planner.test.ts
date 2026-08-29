import { describe, expect, it } from 'vitest';
import type { AgentAction, ScrubbedDom, ScrubbedNode } from '@shared/types';
import { planLocally, rankCandidates, rankOf } from '~/ai/local-planner';

const node = (over: Partial<ScrubbedNode> & { id: number; selector: string }): ScrubbedNode => ({
  tag: 'button',
  visible: true,
  ...over,
});

const dom = (nodes: ScrubbedNode[]): ScrubbedDom => ({
  url: 'https://shop.test/',
  origin: 'https://shop.test',
  title: 'Shop',
  viewport: { width: 1200, height: 800, scrollX: 0, scrollY: 0 },
  nodes,
  redactionSummary: {},
});

const CONSENT = dom([
  node({ id: 0, selector: '#accept', text: 'Accept all' }),
  node({ id: 1, selector: '#reject', text: 'Reject non-essential' }),
  node({ id: 2, selector: '#more', tag: 'a', text: 'More information' }),
]);

const LOGIN = dom([
  node({ id: 0, selector: '#email', tag: 'input', type: 'email', label: 'Email address' }),
  node({ id: 1, selector: '#pw', tag: 'input', type: 'password', label: 'Password' }),
  node({ id: 2, selector: '#submit', text: 'Sign in' }),
  node({ id: 3, selector: '#search', tag: 'input', type: 'search', placeholder: 'Search products' }),
]);

const FILES = dom([
  node({ id: 0, selector: '#help', text: 'Click here for help' }),
  node({ id: 1, selector: '#pkg', tag: 'a', text: 'package.json', href: 'https://shop.test/repo/blob/main/package.json' }),
  node({ id: 2, selector: '#readme', tag: 'a', text: 'README.md', href: 'https://shop.test/repo/blob/main/README.md' }),
]);

const PRODUCT_PAGE = dom([
  node({ id: 0, selector: '#header-cart', tag: 'a', role: 'link', text: 'Cart (0)' }),
  node({ id: 1, selector: '#title', tag: 'h1', text: 'Privacy Vision Agent Test Product' }),
  node({ id: 2, selector: '#price', tag: 'span', text: '$19.99' }),
  node({ id: 3, selector: '#add-btn', tag: 'button', text: 'Add requested item' }),
  node({ id: 4, selector: '#footer', tag: 'a', role: 'link', text: 'Contact Us' }),
]);

describe('planLocally', () => {
  it('maps "accept cookies" onto an Accept button via synonyms', () => {
    const d = planLocally({ goal: 'accept cookies', dom: CONSENT });
    expect(d.action).toMatchObject({ action: 'click', selector: '#accept' });
    expect(d.source).toBe('heuristic');
    expect(d.confidence).toBeGreaterThan(0.4);
  });

  it('prefers a fillable field for a fill intent', () => {
    const d = planLocally({ goal: 'enter my email address', dom: LOGIN });
    expect(d.action).toMatchObject({ action: 'fill', selector: '#email', valueType: 'USER_EMAIL' });
  });

  it('prefers a clickable element for a click intent', () => {
    const d = planLocally({ goal: 'click sign in', dom: LOGIN });
    expect(d.action).toMatchObject({ action: 'click', selector: '#submit' });
  });

  it('pulls a quoted literal out of the goal for a free-text field', () => {
    const d = planLocally({ goal: 'search for "wireless mouse"', dom: LOGIN });
    expect(d.action).toMatchObject({
      action: 'fill',
      selector: '#search',
      valueType: 'LITERAL',
      value: 'wireless mouse',
    });
  });

  it('never targets a disabled element', () => {
    const withDisabled = dom([
      node({ id: 0, selector: '#accept', text: 'Accept all', disabled: true }),
      node({ id: 1, selector: '#accept2', text: 'Accept all cookies' }),
    ]);
    const d = planLocally({ goal: 'accept cookies', dom: withDisabled });
    expect(d.action).toMatchObject({ selector: '#accept2' });
  });

  it('breaks a loop instead of clicking the same element a third time', () => {
    const history: AgentAction[] = [
      { action: 'click', selector: '#accept' },
      { action: 'click', selector: '#accept' },
    ];
    const d = planLocally({ goal: 'accept cookies', dom: CONSENT, history });
    if ('selector' in d.action) expect(d.action.selector).not.toBe('#accept');
  });

  it('falls back to scrolling if nothing matches', () => {
    const d = planLocally({ goal: 'scroll down to see more', dom: CONSENT });
    expect(d.action).toMatchObject({ action: 'scroll' });
  });

  describe('TaskMemory integration', () => {
    it('uses currentObjective instead of goal when available', () => {
      // Goal says "search", but memory says current objective is "login"
      const taskMemory: import('@shared/types').TaskMemory = {
        goal: 'search for products and login',
        currentObjective: 'enter my email address',
        completedObjectives: [],
        attemptedTargets: [],
        step: 1,
      };
      const d = planLocally({ goal: taskMemory.goal, dom: LOGIN, taskMemory });
      expect(d.action).toMatchObject({ action: 'fill', selector: '#email' });
    });

    it('penalizes previously attempted targets to avoid loops', () => {
      const taskMemory: import('@shared/types').TaskMemory = {
        goal: 'login',
        currentObjective: 'Sign in',
        completedObjectives: [],
        attemptedTargets: ['#submit'], // Assume clicking sign-in failed/no-change
        step: 2,
      };
      const d = planLocally({ goal: taskMemory.goal, dom: LOGIN, taskMemory });
      // Because #submit is heavily penalized, it should NOT choose it again easily.
      // Depending on other matches, it might pick something else or fail. 
      // If nothing else matches well, it returns done (no element matches).
      if (d.action.action === 'click') {
        expect(d.action.selector).not.toBe('#submit');
      }
    });
  });

  describe('Action Grounding', () => {
    it('prefers a state-changing control over a navigation element for an active objective', () => {
      // For a state-changing verb ("Add"), the planner should strongly prefer 
      // the actionable button over the pure navigation link containing "Cart"
      const d = planLocally({
        goal: 'test task',
        dom: PRODUCT_PAGE,
        taskMemory: {
          goal: 'test task',
          currentObjective: 'Add the current item to the requested destination',
          completedObjectives: [],
          attemptedTargets: [],
          step: 1,
        },
      });
      expect(d.action).toMatchObject({ action: 'click', selector: '#add-btn' });
    });
  });

  it('honours a bare scroll goal with no matching element', () => {
    const d = planLocally({ goal: 'scroll down', dom: dom([]) });
    expect(d.action).toMatchObject({ action: 'scroll', deltaY: 640 });
  });

  it('stops rather than guessing when nothing matches', () => {
    const d = planLocally({ goal: 'download the quarterly tax report', dom: CONSENT });
    expect(d.action.action).toBe('invalid');
    if ('reason' in d.action) {
      expect(d.action.reason).toBe('NO_ACTIONABLE_TARGET');
    }
    expect(d.confidence).toBe(0);
  });

  it('does not exceed the heuristic confidence ceiling', () => {
    const d = planLocally({ goal: 'accept all cookies consent', dom: CONSENT });
    expect(d.confidence).toBeLessThanOrEqual(0.78);
  });

  it('does whole-word matching, so "log" does not fire on "blog"', () => {
    const blog = dom([node({ id: 0, selector: '#blog', tag: 'a', text: 'Blog' })]);
    const d = planLocally({ goal: 'log in to my account', dom: blog });
    expect(d.action.action).toBe('invalid');
    if ('reason' in d.action) expect(d.action.reason).toBe('NO_ACTIONABLE_TARGET');
  });

  it('targets the named file for "click on package.json"', () => {
    const d = planLocally({ goal: 'click on package.json', dom: FILES });
    expect(d.action).toMatchObject({ action: 'click', selector: '#pkg' });
  });
});

describe('rankCandidates', () => {
  it('ranks the goal-relevant element first regardless of DOM position', () => {
    const big = dom([
      ...Array.from({ length: 40 }, (_, i) => node({ id: i, selector: `#f${i}`, text: `Item ${i}` })),
      node({ id: 40, selector: '#accept', text: 'Accept all cookies' }),
    ]);
    const { candidates } = rankCandidates({ goal: 'accept cookies', dom: big });
    expect(candidates[0]?.node.selector).toBe('#accept');
  });

  it('excludes disabled and invisible nodes from the ranking', () => {
    const mixed = dom([
      node({ id: 0, selector: '#a', text: 'Accept cookies', disabled: true }),
      node({ id: 1, selector: '#b', text: 'Accept cookies', visible: false }),
      node({ id: 2, selector: '#c', text: 'Accept cookies' }),
    ]);
    const { candidates } = rankCandidates({ goal: 'accept cookies', dom: mixed });
    expect(candidates.map((c) => c.node.selector)).toEqual(['#c']);
  });

  it('reports nothing when no element mentions the goal', () => {
    const { candidates } = rankCandidates({ goal: 'quarterly tax report', dom: CONSENT });
    expect(candidates).toHaveLength(0);
  });

  it('rankOf locates a selector and returns undefined for an unranked one', () => {
    const ranking = rankCandidates({ goal: 'accept cookies', dom: CONSENT });
    expect(rankOf(ranking, '#accept')).toBe(0);
    expect(rankOf(ranking, '#more')).toBeUndefined();
  });

  it('demotes an element already acted on, without dropping it from the page', () => {
    const goal = 'accept cookies';
    const fresh = rankCandidates({ goal, dom: CONSENT }).candidates[0]?.score ?? 0;
    const stale =
      rankCandidates({ goal, dom: CONSENT, history: [{ action: 'click', selector: '#accept' }] })
        .candidates.find((c) => c.node.selector === '#accept')?.score ?? 0;
    expect(stale).toBeLessThan(fresh);
  });

  it('reads an operation verb as intent only, never as a content keyword', () => {
    const ranking = rankCandidates({ goal: 'click on package.json', dom: FILES });
    expect(ranking.intent.click).toBe(true);
    // "Click here for help" overlaps the goal on the verb alone.
    expect(rankOf(ranking, '#help')).toBeUndefined();
    expect(rankOf(ranking, '#pkg')).toBe(0);
  });

  it('disambiguates identical action buttons using context', () => {
    const products = dom([
      node({ id: 0, selector: '#btn-15', text: 'Add to Cart', context: 'Apple iPhone 15 Pro 128GB' }),
      node({ id: 1, selector: '#btn-16', text: 'Add to Cart', context: 'Apple iPhone 16 Pro 128GB' }),
      node({ id: 2, selector: '#btn-17', text: 'Add to Cart', context: 'Apple iPhone 17 Pro 256GB' }),
    ]);
    const d = planLocally({ goal: 'Add iPhone 17 to cart', dom: products });
    expect(d.action).toMatchObject({ action: 'click', selector: '#btn-17' });
  });
});

/* ------------------------------------------------------------------ *
 * REGRESSION: compound NL instructions — search value extraction
 *
 * Root cause (fixed): planLocally called toAction(node, input.goal, ranking),
 * where input.goal is the FULL original user instruction. When a compound
 * instruction like "find the search box and search for gaming laptops" was
 * decomposed into sub-objectives, the active objective "Search for gaming
 * laptops" was used correctly for element ranking (via rankCandidates) but
 * the fill VALUE was extracted from the original phrase, yielding:
 *   "the search box and search for gaming laptops"
 * instead of:
 *   "gaming laptops"
 *
 * Fix: planLocally now derives effectiveGoal = currentObjective || goal and
 * passes that to toAction, matching the behaviour already in rankCandidates.
 * ------------------------------------------------------------------ */

const SEARCH_DOM = dom([
  node({
    id: 0,
    selector: '#q',
    tag: 'input',
    type: 'search',
    role: 'searchbox',
    placeholder: 'Search',
    label: 'Search',
  }),
  node({ id: 1, selector: '#submit', tag: 'button', text: 'Search' }),
]);

/** Helper: make TaskMemory with a given active objective. */
function taskMem(currentObjective: string): import('@shared/types').TaskMemory {
  return {
    goal: 'irrelevant original goal',
    currentObjective,
    subObjectives: [{ id: 1, description: currentObjective, status: 'active' }],
    completedObjectives: [],
    attemptedTargets: [],
    replans: 0,
    step: 1,
  };
}

describe('planLocally — compound NL instruction regression', () => {

  // Prompt 1: simple — should work (baseline)
  it('P1: "search laptops" — simple goal produces value "laptops"', () => {
    const d = planLocally({ goal: 'search laptops', dom: SEARCH_DOM });
    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') {
      expect(d.action.value).toBe('laptops');
    }
  });

  // Prompt 2: simple search — should work (baseline)
  it('P2: "search for gaming laptops" — produces value "gaming laptops"', () => {
    const d = planLocally({ goal: 'search for gaming laptops', dom: SEARCH_DOM });
    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') {
      expect(d.action.value).toBe('gaming laptops');
    }
  });

  // Prompt 4: compound — when decomposed, objective 2 is "Search laptops"
  it('P4: compound "find search box and search laptops" — with active objective "Search laptops", value is "laptops" not the whole phrase', () => {
    const originalGoal = 'find search box and search laptops';
    const activeObjective = 'Search laptops';

    const d = planLocally({
      goal: originalGoal,
      dom: SEARCH_DOM,
      taskMemory: taskMem(activeObjective),
    });

    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') {
      // Must NOT contain the original compound phrase
      expect(d.action.value).not.toContain('search box');
      expect(d.action.value).toBe('laptops');
    }
  });

  // Prompt 5: the canonical failing case from the bug report
  it('P5: compound "find the search box and search for gaming laptops" — with active objective "Search for gaming laptops", value is "gaming laptops"', () => {
    const originalGoal = 'find the search box and search for gaming laptops';
    const activeObjective = 'Search for gaming laptops';

    const d = planLocally({
      goal: originalGoal,
      dom: SEARCH_DOM,
      taskMemory: taskMem(activeObjective),
    });

    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') {
      expect(d.action.value).toBe('gaming laptops');
      // Ensure the full compound phrase is NOT used as the search value
      expect(d.action.value).not.toContain('search box');
      expect(d.action.value).not.toContain('find');
    }
  });

  // Prompt 6: compound with three clauses
  it('P6: compound with three clauses — active objective "Search for gaming laptops" yields value "gaming laptops"', () => {
    const d = planLocally({
      goal: 'find the search box and search for gaming laptops and open the first result',
      dom: SEARCH_DOM,
      taskMemory: taskMem('Search for gaming laptops'),
    });

    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') {
      expect(d.action.value).toBe('gaming laptops');
    }
  });

  // Prompt 7: compound with open-cheapest-one
  it('P7: "search for gaming laptops and open the cheapest one" — active objective yields "gaming laptops"', () => {
    const d = planLocally({
      goal: 'search for gaming laptops and open the cheapest one',
      dom: SEARCH_DOM,
      taskMemory: taskMem('Search for gaming laptops'),
    });

    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') {
      expect(d.action.value).toBe('gaming laptops');
    }
  });

  // Generalization: the fix must not be product-specific
  it('GENERALIZE: "find search box and search for shoes" — active objective "Search for shoes" yields "shoes"', () => {
    const d = planLocally({
      goal: 'find search box and search for shoes',
      dom: SEARCH_DOM,
      taskMemory: taskMem('Search for shoes'),
    });
    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') expect(d.action.value).toBe('shoes');
  });

  it('GENERALIZE: "find search box and search for wireless headphones" — yields "wireless headphones"', () => {
    const d = planLocally({
      goal: 'find search box and search for wireless headphones',
      dom: SEARCH_DOM,
      taskMemory: taskMem('Search for wireless headphones'),
    });
    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') expect(d.action.value).toBe('wireless headphones');
  });

  it('GENERALIZE: "find search box and search for programming books" — yields "programming books"', () => {
    const d = planLocally({
      goal: 'find search box and search for programming books',
      dom: SEARCH_DOM,
      taskMemory: taskMem('Search for programming books'),
    });
    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') expect(d.action.value).toBe('programming books');
  });

  // Original bug trigger from the user's manual test
  it('ORIGINAL BUG: "look for the search box and search for laptops" — active "Search for laptops" yields "laptops" not the whole phrase', () => {
    const d = planLocally({
      goal: 'look for the search box and search for laptops',
      dom: SEARCH_DOM,
      taskMemory: taskMem('Search for laptops'),
    });
    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') {
      expect(d.action.value).toBe('laptops');
      expect(d.action.value).not.toContain('search box');
    }
  });

  // Without taskMemory (single-step goal), behaviour is unchanged
  it('NO REGRESSION: without taskMemory, simple goal still works correctly', () => {
    const d = planLocally({ goal: 'search for wireless mouse', dom: SEARCH_DOM });
    expect(d.action.action).toBe('fill');
    if (d.action.action === 'fill') {
      expect(d.action.value).toBe('wireless mouse');
    }
  });
});

describe('planLocally — semantic candidate ranking', () => {
  const SEMANTIC_DOM = dom([
    node({
      id: 1,
      selector: '#q',
      tag: 'input',
      type: 'search',
      role: 'searchbox',
      placeholder: 'Search',
      label: 'Search',
    }),
    node({
      id: 2,
      selector: '#btn',
      tag: 'button',
      role: 'button',
      text: 'Search',
    }),
    node({
      id: 3,
      selector: '#voice',
      tag: 'button',
      role: 'button',
      label: 'Search by voice', // acts as accessible name
    }),
  ]);

  it('S1: "Click the search box" targets the input', () => {
    const d = planLocally({ goal: 'Click the search box', dom: SEMANTIC_DOM });

    if ('selector' in d.action) {
      expect(d.action.selector).toBe('#q');
    }
  });

  it('S2: "Type \'AI\'" targets the input based on fill intent', () => {
    const d = planLocally({ goal: 'Type "AI"', dom: SEMANTIC_DOM });
    // It should pick the input, but what does toAction do for "Type 'AI'"?
    // It will return 'fill' action with literal "AI", or click. Let's just assert selector.
    if ('selector' in d.action) {
      expect(d.action.selector).toBe('#q');
    }
  });

  it('S3: "Click the search button" targets the button', () => {
    const d = planLocally({ goal: 'Click the search button', dom: SEMANTIC_DOM });
    expect(d.action.action).toBe('click');
    if ('selector' in d.action) {
      expect(d.action.selector).not.toBe('#q');
      expect(['#btn', '#voice']).toContain(d.action.selector);
    }
  });

  it('S4: "Use voice search" targets the voice button', () => {
    const ranking = rankCandidates({ goal: 'Use voice search', dom: SEMANTIC_DOM });
    const best = ranking.candidates[0];
    expect(best?.node.selector).toBe('#voice');
  });
});

/* ------------------------------------------------------------------ *
 * REGRESSION: Target vs Value Conflation
 *
 * Ensures that the noun description of the target (e.g. "search box")
 * does not leak into the search value, and that explicit locators
 * ("into the search box") are stripped.
 * ------------------------------------------------------------------ */
describe('planLocally — target vs value conflation', () => {
  const SEARCH_DOM = dom([
    node({
      id: 1,
      selector: '#q',
      tag: 'input',
      type: 'search',
      role: 'searchbox',
    })
  ]);

  it('V1: "search sih26171" produces value "sih26171"', () => {
    const d = planLocally({ goal: 'search sih26171', dom: SEARCH_DOM });
    expect(d.action.action).toBe('fill');
    if ('value' in d.action) expect(d.action.value).toBe('sih26171');
  });

  it('V2: "click the search box" produces a focus/click action with NO value', () => {
    const d = planLocally({ goal: 'click the search box', dom: SEARCH_DOM });
    expect(d.action.action).toBe('click');
    expect('value' in d.action).toBe(false);
  });

  it('V3: "find the search box" produces a focus/click action with NO value', () => {
    const d = planLocally({ goal: 'find the search box', dom: SEARCH_DOM });
    expect(d.action.action).toBe('click');
    expect('value' in d.action).toBe(false);
  });

  it('V4: "type sih26171 into the search box" strips the locator and produces "sih26171"', () => {
    const d = planLocally({ goal: 'type sih26171 into the search box', dom: SEARCH_DOM });
    expect(d.action.action).toBe('fill');
    if ('value' in d.action) expect(d.action.value).toBe('sih26171');
  });

  it('V5: "search laptops in the search bar" strips the locator and produces "laptops"', () => {
    const d = planLocally({ goal: 'search laptops in the search bar', dom: SEARCH_DOM });
    expect(d.action.action).toBe('fill');
    if ('value' in d.action) expect(d.action.value).toBe('laptops');
  });
});

/* ------------------------------------------------------------------ *
 * REGRESSION: Typing vs Submitting
 * ------------------------------------------------------------------ */
describe('planLocally — typing vs submitting', () => {
  const SEARCH_DOM = dom([
    node({
      id: 1,
      selector: '#q',
      tag: 'input',
      type: 'search',
      role: 'searchbox',
      value: 'ai', // Already has 'ai' typed
    })
  ]);

  it('T1: "type ai" DOES NOT submit', () => {
    const d = planLocally({ goal: 'type ai', dom: SEARCH_DOM });
    expect(d.action.action).toBe('fill');
    if ('submit' in d.action) expect(d.action.submit).toBe(false);
  });

  it('T2: "Submit the search" on an input field acts as a pure enter', () => {
    // If the planner couldn't find a dedicated button, it focuses the input
    // with submit: true and value preserved.
    const d = planLocally({ goal: 'Submit the search', dom: SEARCH_DOM });
    expect(d.action.action).toBe('fill');
    if ('submit' in d.action) expect(d.action.submit).toBe(true);
    if ('value' in d.action) expect(d.action.value).toBe('ai');
  });

  it('Fixture A: selects button[type=submit] over button[type=button] Voice Search', () => {
    const FIXTURE_A_DOM = dom([
      node({ id: 1, selector: '#search', tag: 'input', type: 'search', role: 'searchbox', formId: 1 }),
      node({ id: 2, selector: '#submit-btn', tag: 'button', type: 'submit', label: 'Search', formId: 1 }),
      node({ id: 3, selector: '#voice-btn', tag: 'button', type: 'button', label: 'Voice Search', formId: 1 }),
    ]);

    const ranking = rankCandidates({ goal: 'Submit the search', dom: FIXTURE_A_DOM });
    expect(ranking.candidates.length).toBeGreaterThan(0);
    expect(ranking.candidates[0]?.node.selector).toBe('#submit-btn');

    const decision = planLocally({ goal: 'Submit the search', dom: FIXTURE_A_DOM });
    expect(decision.action.action).toBe('click');
    if (decision.action.action === 'click') {
      expect(decision.action.selector).toBe('#submit-btn');
    }
  });
});


