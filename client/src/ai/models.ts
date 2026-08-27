/**
 * Model catalogue. Deliberately free of `@huggingface/transformers` imports
 * so the popup and service worker can read it without pulling ~900 kB of
 * onnxruntime into their bundles.
 */
export type Dtype = 'fp32' | 'fp16' | 'q8' | 'q4' | 'q4f16';

/**
 * SmolVLM/Qwen2-VL load as three ONNX graphs. Quantisation has to be chosen
 * per graph, not once for the whole model:
 *
 *   embed_tokens          — a lookup table. 4-bit rounding here corrupts every
 *                           token embedding, so the decoder emits fluent-looking
 *                           garbage instead of JSON. Keep it fp16/fp32.
 *   vision_encoder        — tolerates q4 well.
 *   decoder_model_merged  — tolerates q4 well; this is where the size lives.
 *
 * A flat `dtype: 'q4'` is what made the local model "load but never decide":
 * it ran, produced no parseable action, and every step fell through to the
 * server.
 */
export type SessionName = 'embed_tokens' | 'vision_encoder' | 'decoder_model_merged';
export type DtypeMap = Record<SessionName, Dtype>;

export interface ModelSpec {
  id: string;
  label: string;
  /** Approximate download size, shown in the popup. */
  sizeMb: number;
  /** Quantisation when a WebGPU adapter is present. */
  webgpu: DtypeMap;
  /** Quantisation for the CPU/wasm fallback (no fp16 there). */
  wasm: DtypeMap;
}

export const MODEL_REGISTRY: Record<string, ModelSpec> = {
  smolvlm256: {
    id: 'HuggingFaceTB/SmolVLM-256M-Instruct',
    label: 'SmolVLM 256M',
    sizeMb: 230,
    webgpu: { embed_tokens: 'fp16', vision_encoder: 'q4', decoder_model_merged: 'q4' },
    wasm: { embed_tokens: 'fp32', vision_encoder: 'q8', decoder_model_merged: 'q8' },
  },
  smolvlm500: {
    id: 'HuggingFaceTB/SmolVLM-500M-Instruct',
    label: 'SmolVLM 500M',
    sizeMb: 420,
    webgpu: { embed_tokens: 'fp16', vision_encoder: 'q4', decoder_model_merged: 'q4' },
    wasm: { embed_tokens: 'fp32', vision_encoder: 'q8', decoder_model_merged: 'q8' },
  },
  qwen2vl2b: {
    id: 'onnx-community/Qwen2-VL-2B-Instruct',
    label: 'Qwen2-VL 2B',
    sizeMb: 1600,
    webgpu: { embed_tokens: 'fp16', vision_encoder: 'q4', decoder_model_merged: 'q4' },
    wasm: { embed_tokens: 'fp32', vision_encoder: 'q8', decoder_model_merged: 'q8' },
  },
};

export const DEFAULT_MODEL_KEY = 'smolvlm256';
