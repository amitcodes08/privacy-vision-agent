/**
 * Routes sub-query work to whichever extension context can actually run
 * Gemini Nano.
 *
 * This file is the load-bearing half of the shift. The orchestrator lives in the
 * service worker, but Chrome exposes the Prompt API to *documents*, not to
 * workers — so calling `decomposeGoal()` from `background/index.ts` probed a
 * global that is not there, and every run silently fell back to clause
 * splitting. The offscreen document is already in the build for the WebGPU
 * worker, and it is a real document, so that is where Nano runs.
 *
 * The route is resolved once per service-worker lifetime:
 *
 *   local     — this context has the Prompt API (a future Chrome exposing it to
 *               extension workers, or when the router is used from a document).
 *   offscreen — delegate over chrome.runtime to the offscreen host.
 *   none      — no Nano anywhere; `decompose` still returns a rule-based plan,
 *               and `replan`/`verify` become no-ops the caller can ignore.
 *
 * Every method degrades instead of throwing. Planning is an optimisation on top
 * of the existing decision ladder; it must never be able to abort a run.
 */
import type { ScrubbedDom } from '@shared/types';
import {
  decomposeGoal,
  nanoStatus,
  replanFromPage,
  verifySubObjective,
  type DecomposedPlan,
  type ReplanInput,
  type ReplanResult,
  type VerifyInput,
  type VerifyResult,
} from '~/ai/nano-query-planner';
import type { NanoProbe } from '~/ai/nano-session';
import { probeNano } from '~/ai/nano-session';

export type NanoRoute = 'local' | 'offscreen' | 'none';

/** Message kinds the offscreen host answers. Mirrored in `offscreen/main.ts`. */
export type NanoMessageKind = 'NANO_PROBE' | 'NANO_DECOMPOSE' | 'NANO_REPLAN' | 'NANO_VERIFY';

type Reply<T> = ({ ok: true } & T) | { ok: false; error: string };

/** Injected transport, so this module needs no import from `background/index.ts`. */
export type NanoDelegate = <T>(msg: { kind: NanoMessageKind } & Record<string, unknown>) => Promise<Reply<T>>;

export interface NanoRouterStatus extends NanoProbe {
  route: NanoRoute;
}

export interface NanoRouter {
  /** Resolve (and cache) where Nano runs. Safe to call repeatedly. */
  route(): Promise<NanoRoute>;
  status(): NanoRouterStatus;
  decompose(goal: string, dom?: ScrubbedDom): Promise<DecomposedPlan>;
  replan(input: ReplanInput): Promise<ReplanResult>;
  verify(input: VerifyInput): Promise<VerifyResult>;
  /** Forget the cached route — used when the offscreen document is torn down. */
  reset(): void;
}

export function createNanoRouter(delegate: NanoDelegate): NanoRouter {
  let resolved: NanoRoute | null = null;
  let resolving: Promise<NanoRoute> | null = null;
  let probe: NanoProbe = { state: 'unavailable', flavour: 'none', reason: 'not probed yet' };

  async function resolve(): Promise<NanoRoute> {
    if (resolved) return resolved;
    resolving ??= (async () => {
      try {
        // 1. This context, if Chrome ever exposes the API to extension workers.
        const here = await probeNano();
        if (here.state === 'available') {
          probe = here;
          return (resolved = 'local');
        }

        // 2. The offscreen document — the documented home for the Prompt API.
        const there = await delegate<{ probe: NanoProbe }>({ kind: 'NANO_PROBE' });
        if (there.ok && there.probe.state === 'available') {
          probe = there.probe;
          return (resolved = 'offscreen');
        }

        // Report the offscreen reason: it is the one that can be acted on
        // (download the model, update Chrome), whereas "not in a worker" cannot.
        probe = there.ok ? there.probe : { state: 'unavailable', flavour: 'none', reason: there.error };
        return (resolved = 'none');
      } finally {
        resolving = null;
      }
    })();
    return resolving;
  }

  return {
    route: resolve,
    status: () => ({ ...probe, route: resolved ?? 'none' }),
    reset() {
      resolved = null;
      resolving = null;
      probe = { state: 'unavailable', flavour: 'none', reason: 'not probed yet' };
    },

    async decompose(goal, dom) {
      if ((await resolve()) === 'offscreen') {
        const r = await delegate<{ plan: DecomposedPlan }>({ kind: 'NANO_DECOMPOSE', goal, dom });
        if (r.ok && r.plan.subObjectives.length > 0) return r.plan;
      }
      // 'local' uses Nano here; 'none' falls through to the rule splitter inside
      // `decomposeGoal`, which is also the right answer for a failed delegation.
      return decomposeGoal(goal, dom);
    },

    async replan(input) {
      if ((await resolve()) === 'offscreen') {
        const r = await delegate<{ result: ReplanResult }>({ kind: 'NANO_REPLAN', input });
        if (r.ok) return r.result;
        return { subObjectives: [...input.remaining], source: 'local-rules', changed: false };
      }
      return replanFromPage(input);
    },

    async verify(input) {
      if ((await resolve()) === 'offscreen') {
        const r = await delegate<{ result: VerifyResult }>({ kind: 'NANO_VERIFY', input });
        if (r.ok) return r.result;
        return { satisfied: false, confidence: 0, reason: r.error, source: 'unavailable' };
      }
      return verifySubObjective(input);
    },
  };
}

/** Re-exported so the service worker can log a local probe without a second import. */
export { nanoStatus };
