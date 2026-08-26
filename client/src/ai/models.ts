/**
 * Model catalogue. Deliberately free of `@huggingface/transformers` imports
 * so the popup and service worker can read it without pulling ~900 kB of
 * onnxruntime into their bundles.
 */
export interface ModelSpec {
  id: string;
  dtype: 'fp32' | 'fp16' | 'q8' | 'q4' | 'q4f16';
  label: string;
  /** Approximate download size, shown in the popup. */
  sizeMb: number;
}

export const MODEL_REGISTRY: Record<string, ModelSpec> = {
  smolvlm256: { id: 'HuggingFaceTB/SmolVLM-256M-Instruct', dtype: 'q4', label: 'SmolVLM 256M (q4)', sizeMb: 230 },
  smolvlm500: { id: 'HuggingFaceTB/SmolVLM-500M-Instruct', dtype: 'q4f16', label: 'SmolVLM 500M (q4f16)', sizeMb: 420 },
  qwen2vl2b: { id: 'onnx-community/Qwen2-VL-2B-Instruct', dtype: 'q4f16', label: 'Qwen2-VL 2B (q4f16)', sizeMb: 1600 },
};

export const DEFAULT_MODEL_KEY = 'smolvlm256';
