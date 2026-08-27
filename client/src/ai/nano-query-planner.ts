/**
 * Client-side query planner and task decomposer.
 *
 * Uses Chrome Built-in AI / Gemini Nano (`ai.languageModel`) when available
 * to break complex, multi-step browser goals into structured, ordered sub-objectives.
 *
 * If Gemini Nano is unavailable or offline, uses a deterministic linguistic
 * clause-decomposer fallback with zero network or download overhead.
 */
import type { TaskObjective } from '@shared/types';

export interface DecomposedPlan {
  subObjectives: TaskObjective[];
  source: 'gemini-nano' | 'local-rules';
}

/** Chrome Prompt API types */
interface ChromeAILanguageModel {
  capabilities?: () => Promise<{ available: 'readily' | 'after-download' | 'no' }>;
  availability?: () => Promise<'readily' | 'after-download' | 'no'>;
  create?: (options?: { systemPrompt?: string; temperature?: number }) => Promise<{
    prompt: (text: string) => Promise<string>;
    destroy?: () => void;
  }>;
}

interface WindowWithAI {
  ai?: {
    languageModel?: ChromeAILanguageModel;
  };
}

const NANO_PROBE_TIMEOUT_MS = 1500;
const NANO_PROMPT_TIMEOUT_MS = 3500;

function withTimeout<T>(promise: Promise<T>, ms: number, fallback: T): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<T>((resolve) => {
    timer = setTimeout(() => resolve(fallback), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/**
 * Checks if Chrome Built-in AI (Gemini Nano) is available in the current environment.
 */
export async function isGeminiNanoAvailable(): Promise<boolean> {
  try {
    const ai = (globalThis as unknown as WindowWithAI).ai?.languageModel;
    if (!ai || typeof ai.create !== 'function') return false;

    if (typeof ai.availability === 'function') {
      const state = await withTimeout(ai.availability(), NANO_PROBE_TIMEOUT_MS, 'no' as const);
      return state === 'readily';
    }

    if (typeof ai.capabilities === 'function') {
      const caps = await withTimeout(ai.capabilities(), NANO_PROBE_TIMEOUT_MS, { available: 'no' as const });
      return caps.available === 'readily';
    }

    return true;
  } catch {
    return false;
  }
}

const SYSTEM_PROMPT = `You are a planner for a web browser automation agent.
Your job is to break down a user's web browsing goal into an ordered sequence of 1 to 6 concise, atomic sub-objectives.
Rules:
- Output ONLY a JSON array of strings: ["sub-goal 1", "sub-goal 2", ...]
- Keep each sub-goal atomic (e.g. "Search for 'laptop'", "Filter by price under $500", "Click on the first search result", "Click 'Add to Cart'")
- If the goal is already simple and single-step (e.g. "Click the login button"), return a single-item array: ["Click the login button"]
- Output NO markdown formatting, NO backticks, ONLY pure valid JSON array.`;

/**
 * Decomposes a user query into ordered sub-objectives.
 */
export async function decomposeGoal(goal: string): Promise<DecomposedPlan> {
  const trimmed = goal.trim();
  if (!trimmed) {
    return { subObjectives: [], source: 'local-rules' };
  }

  // 1. Attempt Chrome Built-in Gemini Nano with bounded timeouts
  try {
    const nanoReady = await isGeminiNanoAvailable();
    if (nanoReady) {
      const ai = (globalThis as unknown as WindowWithAI).ai?.languageModel;
      const session = await withTimeout(
        ai?.create?.({
          systemPrompt: SYSTEM_PROMPT,
          temperature: 0,
        }) ?? Promise.resolve(undefined),
        NANO_PROMPT_TIMEOUT_MS,
        undefined,
      );

      if (session) {
        try {
          const raw = await withTimeout(session.prompt(`Goal: ${trimmed}`), NANO_PROMPT_TIMEOUT_MS, '');
          const parsed = parseJsonArray(raw);
          if (parsed && parsed.length > 0) {
            const subObjectives: TaskObjective[] = parsed.map((desc, idx) => ({
              id: idx + 1,
              description: desc,
              status: idx === 0 ? 'active' : 'pending',
            }));
            return { subObjectives, source: 'gemini-nano' };
          }
        } finally {
          session.destroy?.();
        }
      }
    }
  } catch (err) {
    console.debug('[nano-query-planner] Gemini Nano prompt skipped or failed, using rule decomposer:', err);
  }

  // 2. Deterministic Rule-Based Fallback
  return {
    subObjectives: decomposeWithRules(trimmed),
    source: 'local-rules',
  };
}

/**
 * Deterministic clause-splitting and intent decomposer fallback.
 */
export function decomposeWithRules(goal: string): TaskObjective[] {
  // Regex connectors that separate sequential operations in natural language
  const SPLIT_REGEX = /(?:\s*(?:and\s+then|then|after\s+that|afterwards|and\s+also|followed\s+by|->)\s*|\s*;\s*)/i;

  let clauses = goal.split(SPLIT_REGEX).map((c) => c.trim()).filter(Boolean);

  // If not split by explicit 'then' / 'and then', check for compound 'and' with action verbs
  if (clauses.length === 1) {
    const compoundAndSplit = splitCompoundAnd(goal);
    if (compoundAndSplit.length > 1) {
      clauses = compoundAndSplit;
    }
  }

  // Clean up and format as TaskObjective array
  const subObjectives: TaskObjective[] = clauses
    .map((desc) => normalizeClause(desc))
    .filter(Boolean)
    .map((desc, idx) => ({
      id: idx + 1,
      description: desc,
      status: idx === 0 ? 'active' : 'pending',
    }));

  return subObjectives.length > 0
    ? subObjectives
    : [{ id: 1, description: goal, status: 'active' }];
}

/**
 * Splits compound sentences joined by "and" when followed by an action verb
 * e.g., "Search for shoes and add to cart" -> ["Search for shoes", "Add to cart"]
 */
function splitCompoundAnd(text: string): string[] {
  const ACTION_VERBS = '(?:click|press|open|fill|type|enter|input|select|choose|add|navigate|go\\s+to|visit|check\\s*out|submit|proceed|find|search|scroll)';
  const pattern = new RegExp(`\\s+(?:,\\s*)?and\\s+(?=${ACTION_VERBS}\\b)`, 'i');
  
  const parts = text.split(pattern).map((s) => s.trim()).filter(Boolean);
  return parts.length > 1 ? parts : [text];
}

function normalizeClause(clause: string): string {
  // Strip leading and trailing punctuation and spaces
  const trimmed = clause.replace(/^[;,.\s]+|[;,.\s]+$/g, '');
  if (!trimmed) return '';
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
}

function parseJsonArray(raw: string): string[] | null {
  const start = raw.indexOf('[');
  const end = raw.lastIndexOf(']');
  if (start === -1 || end === -1 || end <= start) return null;

  try {
    const json = JSON.parse(raw.slice(start, end + 1));
    if (Array.isArray(json) && json.every((item) => typeof item === 'string' && item.trim().length > 0)) {
      return json.map((s) => s.trim());
    }
  } catch {
    return null;
  }
  return null;
}
