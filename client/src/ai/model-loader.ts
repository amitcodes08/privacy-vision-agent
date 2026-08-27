/**
 * Local VLM lifecycle: capability probe, model download, and generation.
 * Imported by the worker only — it pulls in onnxruntime.
 *
 * The text -> action logic deliberately lives in `decision-parser.ts` so it can
 * be tested without loading a model.
 */
import {
  AutoProcessor,
  AutoModelForImageTextToText,
  RawImage,
  env,
  type PreTrainedModel,
  type Processor,
  type Tensor,
} from '@huggingface/transformers';
import type { AgentAction, AgentDecision, ScrubbedDom } from '@shared/types';
import { buildPrompt, parseAction } from './decision-parser';
import { rankCandidates } from './local-planner';
import { DEFAULT_MODEL_KEY, MODEL_REGISTRY, type Dtype, type DtypeMap, type ModelSpec } from './models';
import { ORT_WASM_URL, ortWasmConfig } from './ort-assets';

export { DEFAULT_MODEL_KEY, MODEL_REGISTRY, type ModelSpec };
export { buildPrompt, parseAction, extractJson } from './decision-parser';

/**
 * Serve onnxruntime's WASM binary from the extension, and keep onnxruntime on
 * its *embedded* emscripten glue rather than importing any glue at all. Both
 * halves of that matter, and the reasoning lives in `ort-assets.ts` — it is
 * longer than the code.
 */
function useLocalOrtAssets(): void {
  const wasm = (env.backends?.onnx as { wasm?: Record<string, unknown> } | undefined)?.wasm;
  if (!wasm) return;
  Object.assign(wasm, ortWasmConfig());
}

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
    return {
      available: true,
      adapter: describeAdapter(await adapterInfo(adapter)),
      fp16: adapter.features.has('shader-f16'),
    };
  } catch (err) {
    return { available: false, reason: String(err), fp16: false };
  }
}

/**
 * `requestAdapterInfo()` was removed in Chrome 130 in favour of the synchronous
 * `adapter.info`. Reading only the old one is why the log said `webgpu=true
 * unknown` on a machine whose GPU was working fine.
 */
type AdapterInfoish = Partial<Record<'vendor' | 'architecture' | 'device' | 'description', string>>;

async function adapterInfo(adapter: GPUAdapter): Promise<AdapterInfoish | undefined> {
  const a = adapter as GPUAdapter & {
    info?: AdapterInfoish;
    requestAdapterInfo?: () => Promise<AdapterInfoish>;
  };
  if (a.info) return a.info;
  return a.requestAdapterInfo?.().catch(() => undefined);
}

function describeAdapter(info: AdapterInfoish | undefined): string {
  if (!info) return 'adapter details unavailable';
  // Chrome reports these piecemeal and blanks some of them for fingerprinting
  // reasons, so take whatever is actually populated.
  const label = [info.vendor, info.architecture, info.device].filter(Boolean).join(' ').trim();
  return label || info.description?.trim() || 'adapter details unavailable';
}

export interface LoadedModel {
  spec: ModelSpec;
  device: 'webgpu' | 'wasm';
  dtype: DtypeMap;
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

export type ProgressFn = (p: { status: string; file?: string; progress?: number; loaded?: number; total?: number }) => void;

export async function loadModel(key: string, onProgress?: ProgressFn): Promise<LoadedModel> {
  const spec = MODEL_REGISTRY[key] ?? MODEL_REGISTRY[DEFAULT_MODEL_KEY]!;
  const gpu = await probeWebGpu();
  const device: 'webgpu' | 'wasm' = gpu.available ? 'webgpu' : 'wasm';

  // Per-session quantisation. `embed_tokens` must stay wide — see the comment
  // in models.ts. A flat `dtype: 'q4'` also quantises the embedding table,
  // which is what made this model load fine and then never emit a usable
  // action, sending every single step to the escalation server.
  const dtype: DtypeMap =
    device === 'wasm'
      ? spec.wasm
      : gpu.fp16
        ? spec.webgpu
        : { ...spec.webgpu, embed_tokens: 'fp32' as Dtype };

  // Models are fetched from the Hub and cached by the browser; no local
  // model files are bundled with the extension.
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  useLocalOrtAssets();

  onProgress?.({ status: `loading ${spec.label} on ${device}` });
  try {
    const processor = await AutoProcessor.from_pretrained(spec.id, { progress_callback: onProgress as never });

    // `AutoModelForImageTextToText`, not `AutoModelForVision2Seq`: Idefics3 and
    // Qwen2-VL are three-graph (embed_tokens / vision_encoder /
    // decoder_model_merged) image-text-to-text models. Vision2Seq looks for an
    // `encoder_model` that these repos do not ship, and it is the only Auto class
    // Qwen2-VL is not registered under at all.
    const model = await AutoModelForImageTextToText.from_pretrained(spec.id, {
      dtype,
      device,
      progress_callback: onProgress as never,
    });

    onProgress?.({ status: `ready on ${device}`, progress: 100 });
    return { spec, device, dtype, processor, model };
  } catch (err) {
    throw new Error(describeLoadFailure(err, device));
  }
}

/**
 * Turn onnxruntime's opaque load errors into something actionable. The failures
 * below are the ones that actually happen in an MV3 extension, and all of them
 * otherwise surface as an unexplained "model init failed".
 */
function describeLoadFailure(err: unknown, device: string): string {
  const msg = err instanceof Error ? err.message : String(err);
  // Ordered before the WebGPU case deliberately. onnxruntime reports *every*
  // backend failure as `no available backend found. ERR: [webgpu] …`, including
  // ones that happened while instantiating the wasm module — i.e. before WebGPU
  // was reached at all. Matched second, the shader hint below claims these on
  // the word "webgpu" alone and points the reader at the wrong layer.
  if (/no available backend|both async and sync fetching|initializeWebAssembly/i.test(msg)) {
    return `${msg} — onnxruntime never instantiated its wasm module, so no execution provider could start. It expects ${ORT_WASM_URL}: run \`npm run copy:ort\` and reload the extension.`;
  }
  if (/ort-wasm|wasmPaths|Failed to fetch dynamically imported module|import.*\.mjs/i.test(msg)) {
    return `${msg} — onnxruntime's WASM was not found at ${ORT_WASM_URL}. Run \`npm run copy:ort\` and reload the extension.`;
  }
  if (/Content Security Policy|Refused to (load|connect)/i.test(msg)) {
    return `${msg} — blocked by the extension CSP. Check content_security_policy in manifest.json.`;
  }
  if (device === 'webgpu' && /shader|createShaderModule|device.*lost/i.test(msg)) {
    return `${msg} — WebGPU rejected the model's shaders. Try a smaller model, or disable WebGPU to fall back to CPU/WASM.`;
  }
  return msg;
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
  const { loaded, image, goal, dom, history = [], maxNewTokens = 64 } = args;
  const t0 = performance.now();
  const processor = loaded.processor as unknown as ProcessorView;
  const model = loaded.model as unknown as ModelView;

  // One ranking serves both jobs: choosing which elements the model gets to
  // see, and afterwards corroborating whichever one it picked.
  const ranking = rankCandidates({ goal, dom, history });

  const prompt = buildPrompt(goal, dom, history, ranking);
  const messages = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: prompt }] }];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor(text, [image], { do_image_splitting: false });

  const output = await model.generate({ ...inputs, max_new_tokens: maxNewTokens, do_sample: false });

  const raw = decodeCompletion(processor, inputs, output);
  const { action, confidence } = parseAction(raw, dom, { goal, history, ranking });
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
 * Decode only the newly generated tokens.
 *
 * `generate` returns prompt + completion. Decoding the whole thing and then
 * cutting the prompt back off with string matching was unreliable, and when it
 * failed the JSON extractor latched onto the schema example inside the prompt
 * — which parses, but is not a valid action, so the step scored ~0.1 and
 * escalated. Slicing by input length removes that failure mode entirely.
 */
function decodeCompletion(processor: ProcessorView, inputs: Record<string, unknown>, output: unknown): string {
  const promptLen = (inputs.input_ids as Tensor | undefined)?.dims?.at(-1);
  try {
    const out = output as Tensor;
    const total = out.dims?.at(-1);
    if (typeof promptLen === 'number' && typeof total === 'number' && total > promptLen) {
      const decoded = processor.batch_decode(out.slice(null, [promptLen, total]), { skip_special_tokens: true });
      const first = Array.isArray(decoded) ? decoded[0] : undefined;
      if (first !== undefined) return first.trim();
    }
  } catch {
    // Fall through to decoding the full sequence.
  }
  const decoded = processor.batch_decode(output, { skip_special_tokens: true });
  const full = (Array.isArray(decoded) ? (decoded[0] ?? '') : String(decoded)).trim();
  // Last resort: keep only what follows the final assistant turn.
  const at = full.lastIndexOf('Assistant:');
  return at === -1 ? full : full.slice(at + 'Assistant:'.length).trim();
}
