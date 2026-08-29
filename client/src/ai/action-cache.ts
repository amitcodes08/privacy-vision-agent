/**
 * High-performance action and selector cache.
 *
 * Inspired by Stagehand & Midscene.js caching architectures:
 * Fast structural DOM fingerprinting allows repeating verified action sequences
 * without re-querying models or sending network requests when navigating familiar page structures.
 */
import type { AgentAction, AgentDecision, ScrubbedDom, ScrubbedNode } from '@shared/types';

export interface CacheEntry {
  actions: AgentAction[];
  confidence: number;
  ts: number;
  fingerprint: string;
}

const MAX_CACHE_AGE_MS = 30 * 60 * 1000; // 30 minutes
const MAX_ENTRIES_PER_ORIGIN = 50;

export class ActionCache {
  private cache = new Map<string, CacheEntry[]>();
  private hits = 0;

  /**
   * Fast structural fingerprint of interactive DOM nodes.
   * Focuses on element tags, roles, labels, types, and relative structural ordering.
   */
  public computeFingerprint(dom: ScrubbedDom): string {
    const interactive = dom.nodes.filter(
      (n) => n.visible && !n.disabled && (n.role || n.tag === 'input' || n.tag === 'button' || n.tag === 'a' || n.text || n.label),
    );

    const sigs = interactive.slice(0, 40).map((n) => `${n.tag}:${n.type ?? ''}:${n.role ?? ''}:${trunc(n.label || n.text || n.placeholder || '')}`);
    return simpleHash(sigs.join('|'));
  }

  private makeKey(origin: string, objective: string): string {
    const cleanOrigin = origin.toLowerCase().trim();
    const cleanObj = objective.toLowerCase().trim();
    return `${cleanOrigin}::${cleanObj}`;
  }

  public get(origin: string, objective: string, dom: ScrubbedDom): AgentDecision | null {
    const key = this.makeKey(origin, objective);
    const entries = this.cache.get(key);
    if (!entries || entries.length === 0) return null;

    const fp = this.computeFingerprint(dom);
    const now = Date.now();

    const match = entries.find((e) => e.fingerprint === fp && now - e.ts < MAX_CACHE_AGE_MS);
    if (!match) return null;

    // Verify target selectors still exist in the current DOM
    const allSelectorsValid = match.actions.every((act) => {
      if ('selector' in act && act.selector) {
        return this.selectorExistsInDom(act.selector, dom);
      }
      return true;
    });

    if (!allSelectorsValid) {
      // Selector lost validity; drop stale cache entry
      this.invalidate(origin, objective);
      return null;
    }

    this.hits++;
    return {
      action: match.actions[0],
      macroActions: match.actions.length > 1 ? match.actions : undefined,
      confidence: match.confidence,
      source: 'cache',
      latencyMs: 1,
    };
  }

  public set(origin: string, objective: string, dom: ScrubbedDom, actions: AgentAction[], confidence = 0.95): void {
    if (!actions.length) return;
    const key = this.makeKey(origin, objective);
    const fp = this.computeFingerprint(dom);

    const entries = this.cache.get(key) ?? [];
    const filtered = entries.filter((e) => e.fingerprint !== fp);

    filtered.unshift({
      actions,
      confidence,
      ts: Date.now(),
      fingerprint: fp,
    });

    if (filtered.length > MAX_ENTRIES_PER_ORIGIN) {
      filtered.pop();
    }

    this.cache.set(key, filtered);
  }

  public invalidate(origin: string, objective?: string): void {
    if (objective) {
      const key = this.makeKey(origin, objective);
      this.cache.delete(key);
      return;
    }

    const prefix = `${origin.toLowerCase().trim()}::`;
    for (const key of this.cache.keys()) {
      if (key.startsWith(prefix)) {
        this.cache.delete(key);
      }
    }
  }

  public clear(): void {
    this.cache.clear();
    this.hits = 0;
  }

  public getHits(): number {
    return this.hits;
  }

  private selectorExistsInDom(selector: string, dom: ScrubbedDom): boolean {
    if (selector.startsWith('#id-')) {
      const idNum = parseInt(selector.slice(4), 10);
      if (!isNaN(idNum)) {
        return dom.nodes.some((n) => n.id === idNum && n.visible);
      }
    }
    return true;
  }
}

function trunc(s: string, len = 20): string {
  return s.length > len ? s.slice(0, len) : s;
}

function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(36);
}

export const globalActionCache = new ActionCache();
