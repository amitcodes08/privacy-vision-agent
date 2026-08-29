/**
 * Server VLM adapter. Input is already sanitized by the client — this module
 * must never receive or request raw pixels.
 *
 * Supports:
 * 1. Local Ollama VLM (e.g. llama3.2-vision, qwen2-vl, minicpm-v, llava)
 * 2. Google Gemini Cloud VLM (when GOOGLE_API_KEY is provided)
 * 3. Deterministic heuristic planner (offline fallback)
 */
import { GoogleGenAI } from '@google/genai';
import {
  isAgentAction,
  type AgentAction,
  type AgentDecision,
  type InferenceRequestPayload,
  type ScrubbedDom,
} from '../../../shared/types.ts';

export type VlmProvider = 'ollama' | 'gemini' | 'heuristic';

const rawProvider = process.env.VLM_PROVIDER?.toLowerCase();
const apiKey = process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY;

export const OLLAMA_BASE_URL = (process.env.OLLAMA_BASE_URL ?? 'http://127.0.0.1:11434').replace(/\/+$/, '');

/**
 * Hard ceiling on one Ollama call.
 *
 * Cold-loading a 7B vision model takes ~60s, which is longer than the client's
 * own 60s escalation timeout — so without this the client gave up first and
 * the user saw "escalation timed out" while the server was still working, with
 * no fallback ever delivered. Aborting early instead lets the heuristic planner
 * answer within the client's window.
 */
const OLLAMA_TIMEOUT_MS = Number(process.env.OLLAMA_TIMEOUT_MS ?? 25_000);

// Determine active provider:
// Explicit VLM_PROVIDER, or fallback to 'gemini' if API key is present, or default to 'ollama'.
export const provider: VlmProvider = ((): VlmProvider => {
  if (rawProvider === 'gemini' || rawProvider === 'ollama' || rawProvider === 'heuristic') {
    return rawProvider;
  }
  if (apiKey) return 'gemini';
  return 'ollama';
})();

export const MODEL =
  process.env.VLM_MODEL ?? (provider === 'gemini' ? 'gemini-2.5-pro' : 'qwen2.5vl:7b');

const geminiClient = apiKey ? new GoogleGenAI({ apiKey }) : null;

export const activeModelId = ((): string => {
  if (provider === 'ollama') return `ollama/${MODEL}`;
  if (provider === 'gemini' && geminiClient) return `gemini/${MODEL}`;
  return 'heuristic-fallback';
})();

const SYSTEM = `You are the escalation planner for a browser automation agent.
You receive a REDACTED screenshot (black rectangles cover private data) and a
scrubbed element list. Choose exactly ONE next action.

Rules:
- Output JSON only, matching one of these schemas:
  {"action":"click","id":<element number from list>,"reason":"<short>"}
  {"action":"type","id":<element number from list>,"value":"<literal text>","reason":"<short>"}
  {"action":"fill","id":<element number from list>,"valueType":"USER_EMAIL|USER_FULL_NAME|USER_PHONE|USER_ADDRESS|LITERAL","value":"<only when LITERAL>","reason":"<short>"}
  {"action":"select","id":<element number from list>,"value":"<option text>","reason":"<short>"}
  {"action":"scroll","deltaY":<pixels>,"reason":"<short>"}
  {"action":"navigate","url":"<absolute url>"}
  {"action":"back"}
  {"action":"done","summary":"<short>"}
  {"action":"wait","ms":1000}
- "id" must be the NUMBER shown at the start of an ELEMENTS line. Never invent an id.
- Use "type" for non-sensitive literal text (search queries, names). Use "fill" with a valueType for private data.
- IMPORTANT: You must ONLY emit one of the allowed executable actions listed above.
- Disambiguation: When multiple elements have similar actions (e.g. multiple "Add to Cart" buttons)
  inspect each element\'s context, label, or text and choose only the ONE that matches the GOAL.
- Single Execution: Once an action for the active objective/goal has already been performed or
  the goal is satisfied, emit "done" rather than clicking duplicate buttons.
- If the goal requires interacting with a destination, prefer actionable state-changing elements
  (buttons) over navigation links.
- Never invent or guess redacted content. Never emit passwords or OTP codes;
  the client refuses them.
- Prefer "done" when the goal is already satisfied by the visible page.`;

export interface CloudResult {
  decision: AgentDecision;
  rationale?: string;
  usage?: { inputTokens?: number; outputTokens?: number };
}

export async function planAction(req: InferenceRequestPayload): Promise<CloudResult> {
  const started = Date.now();

  if (provider === 'heuristic') {
    const decision = heuristicPlan(req);
    return {
      decision: { ...decision, latencyMs: Date.now() - started },
      rationale: 'heuristic fallback planner',
    };
  }

  if (provider === 'ollama') {
    try {
      return await planWithOllama(req, started);
    } catch (err) {
      console.warn(`[ollama] inference failed (${err instanceof Error ? err.message : String(err)}); falling back to heuristic planner`);
      const fallback = heuristicPlan(req);
      return {
        decision: { ...fallback, latencyMs: Date.now() - started },
        rationale: `Ollama unavailable, fell back to heuristic (${err instanceof Error ? err.message : String(err)})`,
      };
    }
  }

  if (provider === 'gemini') {
    if (!geminiClient) {
      const decision = heuristicPlan(req);
      return {
        decision: { ...decision, latencyMs: Date.now() - started },
        rationale: 'heuristic fallback (no Gemini API key)',
      };
    }
    return planWithGemini(req, started);
  }

  const decision = heuristicPlan(req);
  return { decision: { ...decision, latencyMs: Date.now() - started }, rationale: 'fallback' };
}

/**
 * Inference using local Ollama instance.
 */
async function planWithOllama(req: InferenceRequestPayload, started: number): Promise<CloudResult> {
  // Strip any data-url prefix if present for clean base64 image data
  const base64Data = req.imageBase64.replace(/^data:image\/[a-z]+;base64,/, '');

  const payload = {
    model: MODEL,
    messages: [
      {
        role: 'system',
        content: SYSTEM,
      },
      {
        role: 'user',
        content: userPrompt(req),
        images: [base64Data],
      },
    ],
    format: 'json',
    stream: false,
    options: {
      temperature: 0,
      num_ctx: Number(process.env.OLLAMA_NUM_CTX ?? 32768),
    },
  };

  const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(OLLAMA_TIMEOUT_MS),
  }).catch((err: unknown) => {
    // `AbortError` here means a cold or wedged model, not a bad request. Say so,
    // because "fell back to heuristic" is otherwise mystifying in the logs.
    if (err instanceof Error && err.name === 'TimeoutError') {
      throw new Error(`Ollama did not respond within ${OLLAMA_TIMEOUT_MS}ms (model still loading?)`);
    }
    throw err;
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => response.statusText);
    throw new Error(`Ollama HTTP ${response.status}: ${errText}`);
  }

  const data = (await response.json()) as {
    message?: { content?: string };
    prompt_eval_count?: number;
    eval_count?: number;
  };

  const raw = data.message?.content ?? '';
  const parsed = extractJson(raw) as (Record<string, unknown> & { confidence?: number }) | null;
  const latencyMs = Date.now() - started;

  if (!parsed || !isAgentAction(parsed)) {
    return {
      decision: {
        action: { action: 'done', summary: 'Ollama model returned an unusable plan' },
        confidence: 0,
        source: 'cloud',
        latencyMs,
        modelId: `ollama/${MODEL}`,
      },
      rationale: raw.slice(0, 500),
    };
  }

  const confidence = clamp01(typeof parsed.confidence === 'number' ? parsed.confidence : 0.85);
  return {
    decision: {
      action: sanitize(parsed as AgentAction),
      confidence,
      source: 'cloud',
      latencyMs,
      modelId: `ollama/${MODEL}`,
    },
    rationale: raw.slice(0, 500),
    usage: {
      inputTokens: data.prompt_eval_count,
      outputTokens: data.eval_count,
    },
  };
}

/**
 * Inference using Google Gemini.
 */
async function planWithGemini(req: InferenceRequestPayload, started: number): Promise<CloudResult> {
  const response = await geminiClient!.models.generateContent({
    model: MODEL,
    contents: [
      {
        role: 'user',
        parts: [
          { inlineData: { mimeType: req.imageMime, data: req.imageBase64 } },
          { text: userPrompt(req) },
        ],
      },
    ],
    config: {
      systemInstruction: SYSTEM,
      temperature: 0,
      maxOutputTokens: 400,
      responseMimeType: 'application/json',
    },
  });

  const raw = response.text ?? '';
  const parsed = extractJson(raw) as (Record<string, unknown> & { confidence?: number }) | null;
  const latencyMs = Date.now() - started;

  if (!parsed || !isAgentAction(parsed)) {
    return {
      decision: {
        action: { action: 'done', summary: 'cloud model returned an unusable plan' },
        confidence: 0,
        source: 'cloud',
        latencyMs,
        modelId: `gemini/${MODEL}`,
      },
      rationale: raw.slice(0, 500),
    };
  }

  const confidence = clamp01(typeof parsed.confidence === 'number' ? parsed.confidence : 0.8);
  return {
    decision: {
      action: sanitize(parsed as AgentAction),
      confidence,
      source: 'cloud',
      latencyMs,
      modelId: `gemini/${MODEL}`,
    },
    rationale: raw.slice(0, 500),
    usage: {
      inputTokens: response.usageMetadata?.promptTokenCount,
      outputTokens: response.usageMetadata?.candidatesTokenCount,
    },
  };
}

export function userPrompt(req: InferenceRequestPayload): string {
  const { dom } = req;
  return [
    `GOAL: ${req.goal}`,
    req.taskMemory?.currentObjective ? `CURRENT OBJECTIVE: ${req.taskMemory.currentObjective}` : '',
    req.taskMemory?.completedObjectives?.length ? `COMPLETED OBJECTIVES: ${req.taskMemory.completedObjectives.map(o => o.description).join(', ')}` : '',
    `PAGE: ${dom.title} — ${dom.origin}${dom.url.replace(dom.origin, '')}`,
    `VIEWPORT: ${dom.viewport.width}x${dom.viewport.height} @ scrollY=${dom.viewport.scrollY}`,
    `LOCAL MODEL GAVE UP: confidence=${req.localConfidence ?? 'n/a'} reason=${req.localReason ?? 'n/a'}`,
    req.taskMemory?.lastAction ? `LAST ACTION/RESULT: ${req.taskMemory.lastAction.action.action} -> ${req.taskMemory.lastAction.result}` : '',
    req.history?.length ? `HISTORY: ${req.history.map((h) => h.action).join(' -> ')}` : '',
    'ELEMENTS (id: tag [role=...] [label="..."] [text="..."] [placeholder="..."] [disabled]):',
    ...dom.nodes.slice(0, 80).map(
      (n) =>
        // Render elements by semantic attributes only — no CSS selectors exposed to the cloud model.
        `[${n.id}] <${n.tag}${n.type ? ` type=${n.type}` : ''}>` +
        `${n.role ? ` role=${n.role}` : ''}` +
        `${n.label ? ` label="${n.label}"` : ''}${n.text ? ` text="${n.text}"` : ''}` +
        `${n.placeholder ? ` placeholder="${n.placeholder}"` : ''}` +
        `${n.context ? ` context="${n.context}"` : ''}` +
        `${n.value ? ` value="${n.value}"` : ''}${n.redacted?.length ? ` [redacted:${n.redacted.join(',')}]` : ''}` +
        `${n.disabled ? ' disabled' : ''}`,
    ),
  ]
  .filter(Boolean)
  .join('\n');
}

/** Strip anything the model should not have sent. */
export function sanitize(action: AgentAction): AgentAction {
  if (action.action !== 'fill') return action;
  if (action.valueType === 'USER_PASSWORD' || action.valueType === 'OTP_CODE') {
    return { action: 'done', summary: 'blocked: server attempted a credential fill' };
  }
  if (action.valueType !== 'LITERAL' && 'value' in action) {
    const { value: _dropped, ...rest } = action;
    return rest;
  }
  return action;
}

const STOPWORDS = new Set([
  'the', 'and', 'for', 'with', 'this', 'that', 'then', 'from', 'into', 'your',
  'you', 'please', 'can', 'could', 'would', 'want', 'need', 'get', 'got', 'let',
  'its', 'his', 'her', 'their', 'has', 'have', 'was', 'were', 'are', 'but',
  'not', 'all', 'any', 'out', 'off', 'now', 'page', 'site', 'website', 'button',
  'link', 'field', 'box', 'first', 'next', 'also', 'again', 'more', 'see', 'down', 'up',
  'click', 'clicks', 'clicking', 'press', 'tap', 'hit', 'push', 'type', 'paste',
]);

/**
 * Offline planner: enough signal to demo the loop end to end. Picks the
 * element whose label/text/context best matches the goal's entity and action keywords.
 */
export function heuristicPlan(req: InferenceRequestPayload): AgentDecision {
  const targetText = req.taskMemory?.currentObjective || req.goal;
  const rawWords = targetText.toLowerCase().match(/[a-z0-9]{2,}/g) ?? [];
  const words = rawWords.filter((w) => !STOPWORDS.has(w));
  const activeWords = words.length > 0 ? words : rawWords;

  const attempted = new Set(req.taskMemory?.attemptedTargets ?? []);
  for (const h of req.history ?? []) {
    if ('selector' in h && h.selector) attempted.add(h.selector);
  }

  const score = (n: ScrubbedDom['nodes'][number]) => {
    const hay = `${n.label ?? ''} ${n.text ?? ''} ${n.context ?? ''} ${n.placeholder ?? ''} ${n.name ?? ''}`.toLowerCase();
    let s = 0;
    for (const w of activeWords) {
      if (hay.includes(w)) {
        s += w.length >= 3 ? w.length * 2 : 2;
        if (n.context && n.context.toLowerCase().includes(w)) {
          s += 5; // Context disambiguation bonus
        }
      }
    }
    if (attempted.has(n.selector)) s -= 20; // Loop avoidance
    return s;
  };

  const ranked = [...req.dom.nodes]
    .filter((n) => !n.disabled)
    .map((n) => ({ n, s: score(n) }))
    .sort((a, b) => b.s - a.s);

  const best = ranked[0];
  if (!best || best.s <= 0) {
    return { action: { action: 'done', summary: 'no element matches the goal' }, confidence: 0.2, source: 'heuristic' };
  }
  const node = best.n;
  const fillable = node.tag === 'input' || node.tag === 'textarea';
  const action: AgentAction = fillable
    ? { action: 'fill', selector: node.selector, valueType: node.type === 'email' ? 'USER_EMAIL' : 'USER_FULL_NAME' }
    : { action: 'click', selector: node.selector, reason: `keyword match on "${node.label ?? node.text ?? node.context ?? node.tag}"` };
  return { action, confidence: Math.min(0.75, 0.4 + best.s / 40), source: 'heuristic' };
}

export function extractJson(raw: string): unknown {
  const start = raw.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  for (let i = start; i < raw.length; i++) {
    if (raw[i] === '{') depth++;
    else if (raw[i] === '}' && --depth === 0) {
      try {
        return JSON.parse(raw.slice(start, i + 1));
      } catch {
        return null;
      }
    }
  }
  return null;
}

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
