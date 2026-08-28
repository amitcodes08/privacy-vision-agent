import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  closeNanoPlanner,
  decomposeGoal,
  decomposeWithRules,
  digestPage,
  replanFromPage,
  sanitiseSteps,
  setNanoPlanner,
  verifySubObjective,
} from '~/ai/nano-query-planner';
import type { NanoSession } from '~/ai/nano-session';
import type { ScrubbedDom, TaskObjective } from '@shared/types';

/** A stand-in for a Nano session that replays canned replies in order. */
function fakeSession(replies: string[]): NanoSession & { prompts: string[]; schemas: unknown[] } {
  const prompts: string[] = [];
  const schemas: unknown[] = [];
  let at = 0;
  return {
    flavour: 'language-model',
    prompts,
    schemas,
    async ask(input: string, schema?: object) {
      prompts.push(input);
      schemas.push(schema);
      return replies[at++] ?? '';
    },
    destroy() {},
  };
}

const dom = (over: Partial<ScrubbedDom> = {}): ScrubbedDom => ({
  url: 'https://shop.example.com/search?q=mouse',
  origin: 'https://shop.example.com',
  title: 'Search results',
  viewport: { width: 1280, height: 800, scrollX: 0, scrollY: 0 },
  nodes: [],
  redactionSummary: {},
  ...over,
});

const objective = (description: string, status: TaskObjective['status'] = 'pending'): TaskObjective => ({
  description,
  status,
});

afterEach(() => {
  closeNanoPlanner();
  vi.restoreAllMocks();
});

/* ------------------------------------------------------------------ *
 * Deterministic fallback
 * ------------------------------------------------------------------ */

describe('decomposeWithRules', () => {
  it('decomposes sequential goals separated by "and then"', () => {
    const subObjectives = decomposeWithRules(
      'Search for iPhone 15 and then click the first result and then click Add to Cart',
    );

    expect(subObjectives.map((s) => s.description)).toEqual([
      'Search for iPhone 15',
      'Click the first result',
      'Click Add to Cart',
    ]);
    expect(subObjectives.map((s) => s.status)).toEqual(['active', 'pending', 'pending']);
  });

  it('decomposes sequential goals separated by "then" and semicolons', () => {
    const subObjectives = decomposeWithRules('Go to settings; then update profile picture -> click save');

    expect(subObjectives.map((s) => s.description)).toEqual([
      'Go to settings',
      'Update profile picture',
      'Click save',
    ]);
  });

  it('decomposes compound goals joined by "and" with action verbs', () => {
    const subObjectives = decomposeWithRules('Search for running shoes and add to cart');

    expect(subObjectives.map((s) => s.description)).toEqual(['Search for running shoes', 'Add to cart']);
  });

  it('keeps simple single-step goals as a single sub-objective', () => {
    const subObjectives = decomposeWithRules('Click the login button');

    expect(subObjectives).toHaveLength(1);
    expect(subObjectives[0]?.description).toBe('Click the login button');
    expect(subObjectives[0]?.status).toBe('active');
  });
});

/* ------------------------------------------------------------------ *
 * Decompose
 * ------------------------------------------------------------------ */

describe('decomposeGoal', () => {
  it('falls back to rules when Chrome built-in AI is absent', async () => {
    const res = await decomposeGoal('Search for noise cancelling headphones and add to cart');

    expect(res.source).toBe('local-rules');
    expect(res.subObjectives).toHaveLength(2);
  });

  it('uses Nano steps when a session answers, and marks the first one active', async () => {
    const session = fakeSession(['{"steps":["Click the search box","Type \\"mouse\\"","Click Add to Cart"]}']);
    setNanoPlanner(session);

    const res = await decomposeGoal('buy a mouse');

    expect(res.source).toBe('gemini-nano');
    expect(res.subObjectives.map((s) => s.description)).toEqual([
      'Click the search box',
      'Type "mouse"',
      'Click Add to Cart',
    ]);
    expect(res.subObjectives.map((s) => s.status)).toEqual(['active', 'pending', 'pending']);
  });

  it('constrains the reply with a JSON schema', async () => {
    const session = fakeSession(['{"steps":["Click Sign in"]}']);
    setNanoPlanner(session);

    await decomposeGoal('sign in');

    expect(session.schemas[0]).toMatchObject({ type: 'object', required: ['steps'] });
  });

  it('accepts a bare JSON array from a build that ignored the schema', async () => {
    const session = fakeSession(['```json\n["Click Sign in","Fill the password"]\n```']);
    setNanoPlanner(session);

    const res = await decomposeGoal('log in');

    expect(res.source).toBe('gemini-nano');
    expect(res.subObjectives.map((s) => s.description)).toEqual(['Click Sign in', 'Fill the password']);
  });

  it('falls back to rules when Nano answers with unusable text', async () => {
    setNanoPlanner(fakeSession(['I am sorry, I cannot help with that.']));

    const res = await decomposeGoal('Search for shoes and add to cart');

    expect(res.source).toBe('local-rules');
    expect(res.subObjectives).toHaveLength(2);
  });

  it('falls back to rules when the Nano call rejects', async () => {
    const session: NanoSession = {
      flavour: 'language-model',
      ask: () => Promise.reject(new DOMException('nano call exceeded 9000ms', 'TimeoutError')),
      destroy() {},
    };
    setNanoPlanner(session);

    const res = await decomposeGoal('Search for shoes and then add to cart');

    expect(res.source).toBe('local-rules');
    expect(res.subObjectives).toHaveLength(2);
  });

  it('includes the page digest when a DOM is supplied, and omits it otherwise', async () => {
    const withPage = fakeSession(['{"steps":["Click Add to Cart"]}']);
    setNanoPlanner(withPage);
    await decomposeGoal('add it to the cart', dom({ nodes: [node(1, 'button', { text: 'Add to Cart' })] }));
    expect(withPage.prompts[0]).toContain('CURRENT PAGE:');
    expect(withPage.prompts[0]).toContain('Add to Cart');

    closeNanoPlanner();
    const blind = fakeSession(['{"steps":["Click Add to Cart"]}']);
    setNanoPlanner(blind);
    await decomposeGoal('add it to the cart');
    expect(blind.prompts[0]).not.toContain('CURRENT PAGE:');
  });

  it('returns nothing for an empty goal without calling Nano', async () => {
    const session = fakeSession(['{"steps":["should not be used"]}']);
    setNanoPlanner(session);

    const res = await decomposeGoal('   ');

    expect(res.subObjectives).toEqual([]);
    expect(session.prompts).toHaveLength(0);
  });
});

/* ------------------------------------------------------------------ *
 * sanitiseSteps
 * ------------------------------------------------------------------ */

describe('sanitiseSteps', () => {
  it('strips list numbering, drops non-strings, dedupes, and caps at six', () => {
    expect(
      sanitiseSteps([
        '1. Click search',
        'Step 2) Type mouse',
        'click search', // duplicate of the first, differing only in case
        42,
        null,
        ' ',
        'Open cart',
        'Checkout',
        'Confirm',
        'Pay',
        'Review',
        'Print receipt',
      ]),
    ).toEqual(['Click search', 'Type mouse', 'Open cart', 'Checkout', 'Confirm', 'Pay']);
  });

  it('truncates an over-long step rather than dropping it', () => {
    const [only] = sanitiseSteps(['x'.repeat(400)]);
    expect(only).toHaveLength(120);
  });
});

/* ------------------------------------------------------------------ *
 * Replan
 * ------------------------------------------------------------------ */

describe('replanFromPage', () => {
  const stuck = [objective('Filter by price under $500', 'active'), objective('Add the first result to the cart')];

  it('reports no change when Nano is unavailable', async () => {
    const res = await replanFromPage({
      goal: 'buy a cheap laptop',
      remaining: stuck,
      dom: dom(),
      reason: 'no actionable element',
    });

    expect(res.changed).toBe(false);
    expect(res.subObjectives).toHaveLength(2);
  });

  it('rewrites the remaining work and re-activates the first new step', async () => {
    setNanoPlanner(fakeSession(['{"steps":["Click the \\"Sort by: Price\\" dropdown","Click the first result"]}']));

    const res = await replanFromPage({
      goal: 'buy a cheap laptop',
      remaining: stuck,
      completed: [objective('Search for laptops', 'completed')],
      dom: dom({ nodes: [node(1, 'select', { label: 'Sort by: Price' })] }),
      reason: 'no actionable element',
    });

    expect(res.changed).toBe(true);
    expect(res.source).toBe('gemini-nano-replan');
    expect(res.subObjectives[0]).toMatchObject({ description: 'Click the "Sort by: Price" dropdown', status: 'active' });
    expect(res.subObjectives[1]?.status).toBe('pending');
  });

  it('reports no change when Nano echoes the plan it was given', async () => {
    setNanoPlanner(fakeSession(['{"steps":["Filter by price under $500","Add the first result to the cart"]}']));

    const res = await replanFromPage({
      goal: 'buy a cheap laptop',
      remaining: stuck,
      dom: dom(),
      reason: 'stagnant',
    });

    expect(res.changed).toBe(false);
  });

  it('tells Nano which actions already failed, so it does not re-propose them', async () => {
    const session = fakeSession(['{"steps":["Click Checkout"]}']);
    setNanoPlanner(session);

    await replanFromPage({
      goal: 'check out',
      remaining: stuck,
      dom: dom(),
      history: [{ action: 'click', selector: '#buy-now' }],
      reason: 'repeated with no state change',
    });

    expect(session.prompts[0]).toContain('click(#buy-now)');
    expect(session.prompts[0]).toContain('repeated with no state change');
  });
});

/* ------------------------------------------------------------------ *
 * Verify
 * ------------------------------------------------------------------ */

describe('verifySubObjective', () => {
  it('returns a zero-confidence verdict when Nano is unavailable', async () => {
    const res = await verifySubObjective({ objective: 'Add to cart', dom: dom() });

    expect(res).toMatchObject({ satisfied: false, confidence: 0, source: 'unavailable' });
  });

  it('reads a satisfied verdict, capped below the caller\'s stop threshold', async () => {
    setNanoPlanner(fakeSession(['{"satisfied":true,"reason":"the cart badge shows 1 item"}']));

    const res = await verifySubObjective({
      objective: 'Add to cart',
      dom: dom(),
      lastAction: { action: 'click', selector: '#add' },
    });

    expect(res.satisfied).toBe(true);
    expect(res.source).toBe('gemini-nano');
    // 0.75 is HIGH_CONFIDENCE in termination-checker; a Nano yes must stay under it.
    expect(res.confidence).toBeLessThan(0.75);
    expect(res.reason).toContain('cart badge');
  });

  it('reads a negative verdict', async () => {
    setNanoPlanner(fakeSession(['{"satisfied":false,"reason":"the cart is still empty"}']));

    const res = await verifySubObjective({ objective: 'Add to cart', dom: dom() });

    expect(res).toMatchObject({ satisfied: false, source: 'gemini-nano' });
  });

  it('treats a malformed verdict as no verdict', async () => {
    setNanoPlanner(fakeSession(['{"satisfied":"probably"}']));

    const res = await verifySubObjective({ objective: 'Add to cart', dom: dom() });

    expect(res.source).toBe('unavailable');
    expect(res.confidence).toBe(0);
  });
});

/* ------------------------------------------------------------------ *
 * Page digest — the privacy-relevant part
 * ------------------------------------------------------------------ */

function node(id: number, tag: string, over: Partial<ScrubbedDom['nodes'][number]> = {}): ScrubbedDom['nodes'][number] {
  return { id, tag, selector: `#n${id}`, visible: true, ...over };
}

describe('digestPage', () => {
  it('lists the origin and title but never the full URL', () => {
    const out = digestPage(dom());

    expect(out).toContain('https://shop.example.com');
    expect(out).not.toContain('?q=mouse');
    expect(out).toContain('Search results');
  });

  it('never emits a field value, even an unredacted one', () => {
    const out = digestPage(
      dom({ nodes: [node(1, 'input', { type: 'text', label: 'Coupon', value: 'SPRING50' })] }),
    );

    expect(out).toContain('Coupon');
    expect(out).not.toContain('SPRING50');
  });

  it('describes a redacted node by shape alone', () => {
    const out = digestPage(
      dom({
        nodes: [
          node(1, 'input', {
            type: 'password',
            label: 'Password',
            value: '[REDACTED]',
            redacted: ['password'],
          }),
        ],
      }),
    );

    expect(out).toContain('contents withheld');
    expect(out).not.toContain('Password');
    expect(out).not.toContain('REDACTED');
  });

  it('skips hidden and disabled nodes and reports the count it withheld', () => {
    const nodes = [
      node(1, 'button', { text: 'Visible' }),
      node(2, 'button', { text: 'Hidden', visible: false }),
      ...Array.from({ length: 30 }, (_, i) => node(i + 3, 'a', { text: `link ${i}` })),
    ];
    const out = digestPage(dom({ nodes }));

    expect(out).toContain('Visible');
    expect(out).not.toContain('Hidden');
    expect(out).toMatch(/\(\+\d+ more elements not listed\)/);
  });
});
