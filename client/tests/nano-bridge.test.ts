import { afterEach, describe, expect, it, vi } from 'vitest';
import { createNanoRouter, type NanoDelegate } from '~/ai/nano-bridge';
import { closeNanoPlanner } from '~/ai/nano-query-planner';
import type { ScrubbedDom, TaskObjective } from '@shared/types';

const dom: ScrubbedDom = {
  url: 'https://example.com/',
  origin: 'https://example.com',
  title: 'Example',
  viewport: { width: 800, height: 600, scrollX: 0, scrollY: 0 },
  nodes: [],
  redactionSummary: {},
};

const remaining: TaskObjective[] = [{ description: 'Click Checkout', status: 'active' }];

/**
 * A stand-in for the offscreen document. `handlers` is keyed by message kind;
 * an absent kind answers `{ok:false}`, the same as a torn-down document.
 */
function fakeOffscreen(handlers: Record<string, unknown>): NanoDelegate & { calls: string[] } {
  const calls: string[] = [];
  const delegate = ((msg: { kind: string }) => {
    calls.push(msg.kind);
    const reply = handlers[msg.kind];
    return Promise.resolve(reply ?? { ok: false, error: 'offscreen document not reachable' });
  }) as NanoDelegate & { calls: string[] };
  delegate.calls = calls;
  return delegate;
}

const AVAILABLE = { ok: true, probe: { state: 'available', flavour: 'language-model' } };
const UNAVAILABLE = { ok: true, probe: { state: 'unavailable', flavour: 'none', reason: 'no GPU' } };

afterEach(() => {
  closeNanoPlanner();
  vi.restoreAllMocks();
});

describe('createNanoRouter', () => {
  it('routes to the offscreen document when this context has no Prompt API', async () => {
    const delegate = fakeOffscreen({ NANO_PROBE: AVAILABLE });
    const router = createNanoRouter(delegate);

    expect(await router.route()).toBe('offscreen');
    expect(router.status()).toMatchObject({ route: 'offscreen', state: 'available' });
  });

  it('resolves the route once and reuses it', async () => {
    const delegate = fakeOffscreen({ NANO_PROBE: AVAILABLE });
    const router = createNanoRouter(delegate);

    await Promise.all([router.route(), router.route(), router.route()]);

    expect(delegate.calls.filter((c) => c === 'NANO_PROBE')).toHaveLength(1);
  });

  it('re-probes after reset', async () => {
    const delegate = fakeOffscreen({ NANO_PROBE: AVAILABLE });
    const router = createNanoRouter(delegate);

    await router.route();
    router.reset();
    expect(router.status()).toMatchObject({ route: 'none' });
    await router.route();

    expect(delegate.calls.filter((c) => c === 'NANO_PROBE')).toHaveLength(2);
  });

  it('routes to none when Nano exists nowhere, and surfaces the actionable reason', async () => {
    const router = createNanoRouter(fakeOffscreen({ NANO_PROBE: UNAVAILABLE }));

    expect(await router.route()).toBe('none');
    // The offscreen reason ("no GPU") is the one a user can act on — not
    // "no LanguageModel in this context", which is just where we asked from.
    expect(router.status().reason).toBe('no GPU');
  });

  it('still decomposes with the rule splitter when there is no Nano anywhere', async () => {
    const router = createNanoRouter(fakeOffscreen({ NANO_PROBE: UNAVAILABLE }));

    const plan = await router.decompose('Search for shoes and then add to cart');

    expect(plan.source).toBe('local-rules');
    expect(plan.subObjectives).toHaveLength(2);
  });

  it('delegates decomposition and returns the offscreen plan', async () => {
    const router = createNanoRouter(
      fakeOffscreen({
        NANO_PROBE: AVAILABLE,
        NANO_DECOMPOSE: {
          ok: true,
          plan: {
            source: 'gemini-nano',
            subObjectives: [
              { id: 1, description: 'Click the search box', status: 'active' },
              { id: 2, description: 'Click Add to Cart', status: 'pending' },
            ],
          },
        },
      }),
    );

    const plan = await router.decompose('buy a mouse', dom);

    expect(plan.source).toBe('gemini-nano');
    expect(plan.subObjectives).toHaveLength(2);
  });

  it('falls back to the rule splitter when the offscreen decomposition fails mid-run', async () => {
    // NANO_PROBE answers, NANO_DECOMPOSE does not — the document went away.
    const router = createNanoRouter(fakeOffscreen({ NANO_PROBE: AVAILABLE }));

    const plan = await router.decompose('Search for shoes and then add to cart');

    expect(plan.source).toBe('local-rules');
    expect(plan.subObjectives).toHaveLength(2);
  });

  it('delegates a re-plan', async () => {
    const router = createNanoRouter(
      fakeOffscreen({
        NANO_PROBE: AVAILABLE,
        NANO_REPLAN: {
          ok: true,
          result: {
            changed: true,
            source: 'gemini-nano-replan',
            subObjectives: [{ id: 1, description: 'Click Place Order', status: 'active' }],
          },
        },
      }),
    );

    const res = await router.replan({ goal: 'check out', remaining, dom, reason: 'stagnant' });

    expect(res).toMatchObject({ changed: true, source: 'gemini-nano-replan' });
  });

  it('reports "no change" — never a throw — when a delegated re-plan fails', async () => {
    const router = createNanoRouter(fakeOffscreen({ NANO_PROBE: AVAILABLE }));

    const res = await router.replan({ goal: 'check out', remaining, dom, reason: 'stagnant' });

    expect(res.changed).toBe(false);
    expect(res.subObjectives).toEqual(remaining);
  });

  it('delegates verification', async () => {
    const router = createNanoRouter(
      fakeOffscreen({
        NANO_PROBE: AVAILABLE,
        NANO_VERIFY: { ok: true, result: { satisfied: true, confidence: 0.7, reason: 'cart shows 1', source: 'gemini-nano' } },
      }),
    );

    expect(await router.verify({ objective: 'Add to cart', dom })).toMatchObject({ satisfied: true });
  });

  it('reports an unsatisfied, zero-confidence verdict when verification fails', async () => {
    const router = createNanoRouter(fakeOffscreen({ NANO_PROBE: AVAILABLE }));

    const res = await router.verify({ objective: 'Add to cart', dom });

    expect(res).toMatchObject({ satisfied: false, confidence: 0, source: 'unavailable' });
  });
});
