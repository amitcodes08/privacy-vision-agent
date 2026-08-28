import { afterEach, describe, expect, it, vi } from 'vitest';
import { NANO_TIMEOUTS, openNanoSession, probeNano } from '~/ai/nano-session';

/* ------------------------------------------------------------------ *
 * Fake Chrome Prompt API scopes
 * ------------------------------------------------------------------ */

interface FakeSpy {
  createOptions: Record<string, unknown>[];
  promptCalls: { input: string; opts?: Record<string, unknown> }[];
  destroyed: number;
  clones: number;
}

interface FakeOpts {
  /** Modern `LanguageModel` global. */
  modern?: {
    availability?: string | (() => Promise<string>);
    params?: Record<string, number> | null | (() => Promise<never>);
    reply?: string;
    /** Reject a prompt that carries `responseConstraint`. */
    rejectSchema?: boolean;
    clone?: boolean;
    /** Never settle; only resolve/reject via the abort signal. */
    hang?: boolean;
  };
  /** Legacy origin-trial `ai.languageModel`. */
  legacy?: { availability?: string; capabilities?: Record<string, unknown>; reply?: string };
}

function makeScope(opts: FakeOpts): { scope: object; spy: FakeSpy } {
  const spy: FakeSpy = { createOptions: [], promptCalls: [], destroyed: 0, clones: 0 };
  const scope: Record<string, unknown> = {};

  const session = (reply: string, cfg: FakeOpts['modern'] = {}) => {
    const self: Record<string, unknown> = {
      prompt: (input: string, o?: Record<string, unknown>) => {
        spy.promptCalls.push({ input, opts: o });
        const signal = o?.signal as AbortSignal | undefined;
        if (cfg.hang) {
          return new Promise<string>((_res, rej) => {
            signal?.addEventListener('abort', () => rej(signal.reason ?? new Error('aborted')));
          });
        }
        if (cfg.rejectSchema && o?.responseConstraint) {
          return Promise.reject(new TypeError('responseConstraint is not supported'));
        }
        return Promise.resolve(reply);
      },
      destroy: () => {
        spy.destroyed++;
      },
    };
    if (cfg.clone) {
      self.clone = () => {
        spy.clones++;
        return Promise.resolve(session(reply, { ...cfg, clone: false }));
      };
    }
    return self;
  };

  if (opts.modern) {
    const m = opts.modern;
    scope.LanguageModel = {
      availability:
        typeof m.availability === 'function'
          ? m.availability
          : () => Promise.resolve(m.availability ?? 'available'),
      params:
        typeof m.params === 'function'
          ? m.params
          : () => Promise.resolve(m.params === null ? null : (m.params ?? { defaultTopK: 3, maxTopK: 128 })),
      create: (o?: Record<string, unknown>) => {
        spy.createOptions.push(o ?? {});
        return Promise.resolve(session(m.reply ?? '{"ok":true}', m));
      },
    };
  }

  if (opts.legacy) {
    const l = opts.legacy;
    scope.ai = {
      languageModel: {
        ...(l.availability ? { availability: () => Promise.resolve(l.availability!) } : {}),
        ...(l.capabilities ? { capabilities: () => Promise.resolve(l.capabilities!) } : {}),
        create: (o?: Record<string, unknown>) => {
          spy.createOptions.push(o ?? {});
          return Promise.resolve(session(l.reply ?? '{"ok":true}'));
        },
      },
    };
  }

  return { scope, spy };
}

afterEach(() => {
  vi.useRealTimers();
});

/* ------------------------------------------------------------------ *
 * probeNano
 * ------------------------------------------------------------------ */

describe('probeNano', () => {
  it('reports unavailable in a context with no Prompt API — the service-worker case', async () => {
    const probe = await probeNano({});

    expect(probe).toMatchObject({ state: 'unavailable', flavour: 'none' });
    expect(probe.reason).toMatch(/no LanguageModel/);
  });

  it('reads the modern LanguageModel global', async () => {
    const { scope } = makeScope({ modern: { availability: 'available' } });

    expect(await probeNano(scope)).toMatchObject({ state: 'available', flavour: 'language-model' });
  });

  it('translates the legacy availability vocabulary', async () => {
    const readily = makeScope({ legacy: { availability: 'readily' } });
    expect(await probeNano(readily.scope)).toMatchObject({ state: 'available', flavour: 'legacy-ai' });

    const afterDownload = makeScope({ legacy: { availability: 'after-download' } });
    expect(await probeNano(afterDownload.scope)).toMatchObject({ state: 'downloadable' });

    const no = makeScope({ legacy: { availability: 'no' } });
    expect(await probeNano(no.scope)).toMatchObject({ state: 'unavailable' });
  });

  it('falls back to capabilities() when availability() is absent', async () => {
    const { scope } = makeScope({ legacy: { capabilities: { available: 'readily' } } });

    expect(await probeNano(scope)).toMatchObject({ state: 'available', flavour: 'legacy-ai' });
  });

  it('prefers the modern global when both are present', async () => {
    const { scope } = makeScope({ modern: {}, legacy: { availability: 'readily' } });

    expect(await probeNano(scope)).toMatchObject({ flavour: 'language-model' });
  });

  it('falls through to the legacy global when the modern one reports unavailable', async () => {
    const { scope } = makeScope({ modern: { availability: 'unavailable' }, legacy: { availability: 'readily' } });

    expect(await probeNano(scope)).toMatchObject({ state: 'available', flavour: 'legacy-ai' });
  });

  it('reports unavailable, not a throw, when availability() rejects', async () => {
    const { scope } = makeScope({
      modern: { availability: () => Promise.reject(new Error('not allowed in this context')) },
    });

    const probe = await probeNano(scope);
    expect(probe.state).toBe('unavailable');
    expect(probe.reason).toMatch(/not allowed in this context/);
  });

  it('gives up on an availability() call that hangs', async () => {
    vi.useFakeTimers();
    const { scope } = makeScope({ modern: { availability: () => new Promise<string>(() => {}) } });

    const pending = probeNano(scope);
    await vi.advanceTimersByTimeAsync(NANO_TIMEOUTS.probeMs + 50);

    expect((await pending).state).toBe('unavailable');
  });
});

/* ------------------------------------------------------------------ *
 * openNanoSession
 * ------------------------------------------------------------------ */

describe('openNanoSession', () => {
  it('returns null rather than throwing when there is no Prompt API', async () => {
    expect(await openNanoSession({ system: 'plan' }, {})).toBeNull();
  });

  it('refuses to trigger a weight download unless the caller allows it', async () => {
    const { scope, spy } = makeScope({ modern: { availability: 'downloadable' } });

    expect(await openNanoSession({ system: 'plan' }, scope)).toBeNull();
    expect(spy.createOptions).toHaveLength(0);

    expect(await openNanoSession({ system: 'plan' }, scope, { allowDownload: true })).not.toBeNull();
    expect(spy.createOptions).toHaveLength(1);
  });

  it('sends the system instruction as initialPrompts on the modern API', async () => {
    const { scope, spy } = makeScope({ modern: {} });

    await openNanoSession({ system: 'you are a planner' }, scope);

    expect(spy.createOptions[0]).toMatchObject({
      initialPrompts: [{ role: 'system', content: 'you are a planner' }],
    });
    expect(spy.createOptions[0]).not.toHaveProperty('systemPrompt');
  });

  it('sends it as systemPrompt on the legacy API', async () => {
    const { scope, spy } = makeScope({ legacy: { availability: 'readily' } });

    await openNanoSession({ system: 'you are a planner' }, scope);

    expect(spy.createOptions[0]).toMatchObject({ systemPrompt: 'you are a planner' });
    expect(spy.createOptions[0]).not.toHaveProperty('initialPrompts');
  });

  it('pairs temperature with a topK from params(), because extensions reject one alone', async () => {
    const { scope, spy } = makeScope({ modern: { params: { defaultTopK: 5 } } });

    await openNanoSession({ system: 'plan' }, scope);

    expect(spy.createOptions[0]).toMatchObject({ temperature: 0, topK: 5 });
  });

  it('passes neither sampling knob when params() gives no topK', async () => {
    const { scope, spy } = makeScope({ modern: { params: null } });

    await openNanoSession({ system: 'plan' }, scope);

    expect(spy.createOptions[0]).not.toHaveProperty('temperature');
    expect(spy.createOptions[0]).not.toHaveProperty('topK');
  });

  it('constrains the reply with the schema it is given', async () => {
    const { scope, spy } = makeScope({ modern: { reply: '{"steps":["a"]}' } });
    const session = await openNanoSession({ system: 'plan' }, scope);

    const out = await session!.ask('go', { type: 'object' });

    expect(out).toBe('{"steps":["a"]}');
    expect(spy.promptCalls[0]?.opts).toMatchObject({ responseConstraint: { type: 'object' } });
  });

  it('retries without the constraint when the build rejects responseConstraint', async () => {
    const { scope, spy } = makeScope({ modern: { reply: '["a"]', rejectSchema: true } });
    const session = await openNanoSession({ system: 'plan' }, scope);

    expect(await session!.ask('go', { type: 'object' })).toBe('["a"]');
    expect(spy.promptCalls).toHaveLength(2);
    expect(spy.promptCalls[1]?.opts).not.toHaveProperty('responseConstraint');
  });

  it('clones per query so one run\'s planning calls do not accumulate context', async () => {
    const { scope, spy } = makeScope({ modern: { clone: true } });
    const session = await openNanoSession({ system: 'plan' }, scope);

    await session!.ask('first');
    await session!.ask('second');

    expect(spy.clones).toBe(2);
    // Both clones torn down; the root is still alive.
    expect(spy.destroyed).toBe(2);
  });

  it('works against a factory with no clone()', async () => {
    const { scope, spy } = makeScope({ modern: { clone: false, reply: 'ok' } });
    const session = await openNanoSession({ system: 'plan' }, scope);

    expect(await session!.ask('go')).toBe('ok');
    expect(spy.destroyed).toBe(0);
  });

  it('aborts — not merely ignores — a generation that overruns', async () => {
    vi.useFakeTimers();
    const { scope } = makeScope({ modern: { hang: true } });
    const session = await openNanoSession({ system: 'plan' }, scope);

    const pending = session!.ask('go');
    const assertion = expect(pending).rejects.toThrow(/exceeded/);
    await vi.advanceTimersByTimeAsync(NANO_TIMEOUTS.promptMs + 50);
    await assertion;
  });

  it('rejects a query issued after destroy() instead of resurrecting the session', async () => {
    const { scope, spy } = makeScope({ modern: {} });
    const session = await openNanoSession({ system: 'plan' }, scope);

    session!.destroy();
    session!.destroy(); // idempotent

    await expect(session!.ask('go')).rejects.toThrow(/destroyed/);
    expect(spy.destroyed).toBe(1);
  });
});
