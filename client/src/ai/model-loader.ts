/**
 * Local VLM lifecycle: capability probe, model download, generation, and
 * the confidence score that gates escalation. Imported by the worker only.
 */
import {
  AutoProcessor,
  AutoModelForVision2Seq,
  RawImage,
  env,
  type PreTrainedModel,
  type Processor,
} from '@huggingface/transformers';
import { isAgentAction, type AgentAction, type AgentDecision, type ScrubbedDom } from '@shared/types';
import { DEFAULT_MODEL_KEY, MODEL_REGISTRY, type ModelSpec } from './models';

export { DEFAULT_MODEL_KEY, MODEL_REGISTRY, type ModelSpec };

export interface WebGpuReport {
  available: boolean;
  adapter?: string;
  reason?: string;
  fp16: boolean;
}

/** Never throws — a missing adapter must degrade to escalation, not crash. */
export async function probeWebGpu(): Promise<WebGpuReport> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return { available: false, reason: 'navigator.gpu undefined', fp16: false };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { available: false, reason: 'no WebGPU adapter', fp16: false };
    const info = await (adapter as GPUAdapter & { requestAdapterInfo?: () => Promise<GPUAdapterInfo> })
      .requestAdapterInfo?.()
      .catch(() => undefined);
    return {
      available: true,
      adapter: info ? `${info.vendor ?? '?'} ${info.architecture ?? ''}`.trim() : 'unknown',
      fp16: adapter.features.has('shader-f16'),
    };
  } catch (err) {
    return { available: false, reason: String(err), fp16: false };
  }
}

export interface LoadedModel {
  spec: ModelSpec;
  device: 'webgpu' | 'wasm';
  processor: Processor;
  model: PreTrainedModel;
}

/**
 * Transformers.js types the processor call signature loosely across model
 * families; these views keep the casts in one place instead of at each use.
 */
type ProcessorView = {
  (text: string, images: RawImage[], opts?: Record<string, unknown>): Promise<Record<string, unknown>>;
  apply_chat_template: (messages: unknown, opts?: Record<string, unknown>) => string;
  batch_decode: (tensor: unknown, opts?: Record<string, unknown>) => string[];
};
type ModelView = { generate: (args: Record<string, unknown>) => Promise<unknown> };

export type ProgressFn = (p: { status: string; file?: string; progress?: number }) => void;

export async function loadModel(key: string, onProgress?: ProgressFn): Promise<LoadedModel> {
  const spec = MODEL_REGISTRY[key] ?? MODEL_REGISTRY[DEFAULT_MODEL_KEY]!;
  const gpu = await probeWebGpu();
  const device: 'webgpu' | 'wasm' = gpu.available ? 'webgpu' : 'wasm';
  const dtype = device === 'wasm' ? 'q8' : gpu.fp16 ? spec.dtype : 'q4';

  // Models are fetched from the Hub and cached by the browser; no local
  // model files are bundled with the extension.
  env.allowLocalModels = false;
  env.useBrowserCache = true;

  onProgress?.({ status: `loading ${spec.label} on ${device}` });
  const processor = await AutoProcessor.from_pretrained(spec.id, { progress_callback: onProgress as never });
  const model = await AutoModelForVision2Seq.from_pretrained(spec.id, {
    dtype,
    device,
    progress_callback: onProgress as never,
  });
  return { spec, device, processor, model };
}

/** Compact, token-cheap page description for the local model. */
export function buildPrompt(goal: string, dom: ScrubbedDom, history: AgentAction[] = []): string {
  const lines = dom.nodes.slice(0, 60).map((n) => {
    const bits = [n.tag, n.type && `type=${n.type}`, n.label && `"${n.label}"`, n.text && `text="${n.text}"`, n.value && `value="${n.value}"`]
      .filter(Boolean)
      .join(' ');
    return `[${n.id}] ${bits} -> ${n.selector}`;
  });
  const past = history.slice(-3).map((h) => JSON.stringify(h)).join(' ');
  return [
    'You control a web browser. Decide the single next UI action.',
    `GOAL: ${goal}`,
    `PAGE: ${dom.title} (${dom.origin})`,
    'ELEMENTS:',
    lines.join('\n'),
    past && `RECENT: ${past}`,
    'Reply with ONE JSON object only, no prose:',
    '{"action":"click|fill|scroll|navigate|wait|done|escalate","selector":"<css>","valueType":"USER_EMAIL|USER_FULL_NAME|USER_PHONE|LITERAL","value":"<only if LITERAL>"}',
    'Use {"action":"escalate"} when the screenshot or elements are not enough to be sure.',
  ]
    .filter(Boolean)
    .join('\n');
}

export interface GenerateArgs {
  loaded: LoadedModel;
  image: RawImage;
  goal: string;
  dom: ScrubbedDom;
  history?: AgentAction[];
  maxNewTokens?: number;
}

export async function generateDecision(args: GenerateArgs): Promise<AgentDecision & { raw: string }> {
  const { loaded, image, goal, dom, history = [], maxNewTokens = 96 } = args;
  const t0 = performance.now();
  const processor = loaded.processor as unknown as ProcessorView;
  const model = loaded.model as unknown as ModelView;

  const prompt = buildPrompt(goal, dom, history);
  const messages = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: prompt }] }];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor(text, [image], { do_image_splitting: false });

  const output = await model.generate({ ...inputs, max_new_tokens: maxNewTokens, do_sample: false });
  const decoded = processor.batch_decode(output, { skip_special_tokens: true });
  const raw = Array.isArray(decoded) ? (decoded[0] ?? '') : String(decoded);
  const { action, confidence } = parseAction(stripPrompt(raw, prompt), dom);
  return {
    action,
    confidence,
    source: 'local',
    latencyMs: Math.round(performance.now() - t0),
    modelId: loaded.spec.id,
    raw,
  };
}

/**
 * Parse the model's text into an action and score how much we trust it.
 * The score is structural, not probabilistic: a selector that does not
 * exist on the page is worthless no matter how confident the logits were.
 */
export function parseAction(raw: string, dom: ScrubbedDom): { action: AgentAction; confidence: number } {
  const json = extractJson(raw);
  if (!json) return { action: { action: 'escalate', reason: 'no JSON in local output' }, confidence: 0 };
  if (!isAgentAction(json)) return { action: { action: 'escalate', reason: 'schema mismatch' }, confidence: 0.1 };

  let confidence = 0.55;
  const selector = (json as { selector?: string }).selector;
  if (selector) {
    const known = dom.nodes.find((n) => n.selector === selector);
    if (known) confidence += 0.3;
    else confidence -= 0.35;
    if (known?.disabled) confidence -= 0.2;
  }
  if (json.action === 'escalate') confidence = 0;
  if (json.action === 'fill' && !json.valueType) confidence -= 0.2;
  // Repeated boilerplate or truncated JSON is a tell for a confused model.
  if (raw.length > 600) confidence -= 0.1;
  return { action: json, confidence: Math.max(0, Math.min(1, confidence)) };
}

/**
 * `batch_decode` returns the prompt plus the completion for most VLM heads.
 * Drop the echoed prompt so JSON extraction cannot latch onto the example
 * object we included in the instructions.
 */
export function stripPrompt(decoded: string, prompt: string): string {
  const tail = prompt.slice(-64);
  const at = decoded.lastIndexOf(tail);
  return at === -1 ? decoded : decoded.slice(at + tail.length);
}

export function extractJson(raw: string): unknown {
  const start = raw.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  for (let i = start; i < raw.length; i++) {
    if (raw[i] === '{') depth++;
    else if (raw[i] === '}') {
      depth--;
      if (depth === 0) {
        try {
          return JSON.parse(raw.slice(start, i + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}
