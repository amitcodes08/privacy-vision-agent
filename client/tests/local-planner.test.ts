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

