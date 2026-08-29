/**
 * Speculative Sub-Query Pipelining Engine.
 *
 * Inspired by speculative execution in high-performance computing and
 * predictive web agent architectures (UI-TARS, Skim, Browser-Use):
 *
 * While the current action is executing, this module speculatively pre-grounds
 * the NEXT pending sub-objective against the current DOM snapshot. If the current
 * action completes without navigating to an entirely new page, the pre-grounded
 * speculative action can be dispatched immediately with zero model latency.
 */
import type { AgentAction, AgentDecision, ScrubbedDom, TaskMemory, TaskObjective } from '@shared/types';
import { planLocally } from './local-planner';
import { globalActionCache } from './action-cache';

export interface SpeculativePlan {
  objective: string;
  decision: AgentDecision;
  domFingerprint: string;
  ts: number;
}

export class SpeculativePlanner {
  private currentSpeculation: SpeculativePlan | null = null;

  /**
   * Speculatively pre-grounds the next pending sub-objective in background.
   */
  public speculateNext(
    taskMemory: TaskMemory,
    dom: ScrubbedDom,
    history: AgentAction[] = [],
  ): SpeculativePlan | null {
    const subObjectives = taskMemory.subObjectives ?? [];
    if (subObjectives.length === 0) return null;

    // Find the next pending sub-objective after the active one
    const activeIndex = subObjectives.findIndex(
      (o) => o.status === 'active' || o.description === taskMemory.currentObjective,
    );

    const nextObjective = subObjectives.find(
      (o, idx) => idx > activeIndex && o.status === 'pending',
    );

    if (!nextObjective) {
      this.currentSpeculation = null;
      return null;
    }

    const speculativeMemory: TaskMemory = {
      ...taskMemory,
      currentObjective: nextObjective.description,
    };

    const planned = planLocally({
      goal: nextObjective.description,
      dom,
      history,
      taskMemory: speculativeMemory,
    });

    if (
      planned &&
      planned.confidence >= 0.75 &&
      planned.action.action !== 'invalid' &&
      planned.action.action !== 'escalate' &&
      planned.action.action !== 'done'
    ) {
      const fp = globalActionCache.computeFingerprint(dom);
      this.currentSpeculation = {
        objective: nextObjective.description,
        decision: planned,
        domFingerprint: fp,
        ts: Date.now(),
      };
      return this.currentSpeculation;
    }

    this.currentSpeculation = null;
    return null;
  }

  /**
   * Attempts to consume a pre-grounded speculative action if the objective matches
   * and the page structure is still compatible.
   */
  public consume(
    activeObjective: string,
    currentDom: ScrubbedDom,
  ): AgentDecision | null {
    if (!this.currentSpeculation) return null;

    const { objective, decision, domFingerprint, ts } = this.currentSpeculation;
    const isRecent = Date.now() - ts < 15_000; // 15s validity window
    const objMatches =
      objective.toLowerCase().trim() === activeObjective.toLowerCase().trim();

    if (!isRecent || !objMatches) {
      this.currentSpeculation = null;
      return null;
    }

    const currentFp = globalActionCache.computeFingerprint(currentDom);
    if (currentFp !== domFingerprint) {
      this.currentSpeculation = null;
      return null;
    }

    // Verify selector is still present in current DOM
    if ('selector' in decision.action && decision.action.selector) {
      const exists = currentDom.nodes.some(
        (n) => n.selector === (decision.action as any).selector && n.visible,
      );
      if (!exists) {
        this.currentSpeculation = null;
        return null;
      }
    }

    this.currentSpeculation = null; // Single-use consumption
    return {
      ...decision,
      source: 'heuristic',
      latencyMs: 0,
    };
  }

  public clear(): void {
    this.currentSpeculation = null;
  }
}

export const globalSpeculativePlanner = new SpeculativePlanner();
