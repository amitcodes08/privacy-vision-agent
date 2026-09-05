import {
  pipeline,
  env,
} from '@huggingface/transformers';
import { probeWebGpu } from './model-loader';
import { ortWasmConfig } from './ort-assets';

export const STT_MODEL_ID = 'onnx-community/whisper-tiny.en';
export const STT_MODEL_LABEL = 'Whisper tiny.en';
export const STT_SIZE_MB = 39;

type WhisperPipeline = (audio: Float32Array, opts?: Record<string, unknown>) => Promise<{ text: string } | Array<{ text: string }>>;

export interface LoadedSttModel {
  pipe: WhisperPipeline;
  device: 'webgpu' | 'wasm';
}

export type SttProgressFn = (p: {
  status: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}) => void;

function useLocalOrtAssets(): void {
  const wasm = (env.backends?.onnx as { wasm?: Record<string, unknown> } | undefined)?.wasm;
  if (!wasm) return;
  Object.assign(wasm, ortWasmConfig());
}

function cleanRepeatedTokens(text: string): string {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length <= 1) return text;
  const cleaned: string[] = [];
  let repeatCount = 0;
  for (let i = 0; i < words.length; i++) {
    const w = words[i]!;
    if (cleaned.length > 0 && cleaned[cleaned.length - 1] === w) {
      repeatCount++;
      if (repeatCount < 2) {
        cleaned.push(w);
      }
    } else {
      repeatCount = 0;
      cleaned.push(w);
    }
  }
  return cleaned.join(' ');
}

export async function loadSttModel(onProgress?: SttProgressFn): Promise<LoadedSttModel> {
  const gpu = await probeWebGpu();
  const device: 'webgpu' | 'wasm' = gpu.available ? 'webgpu' : 'wasm';

  const dtype =
    device === 'webgpu'
      ? { encoder_model: 'fp32' as const, decoder_model_merged: 'q4' as const }
      : 'q8' as const;

  env.allowLocalModels = false;
  env.useBrowserCache = true;
  useLocalOrtAssets();

  onProgress?.({ status: `loading ${STT_MODEL_LABEL} on ${device}` });

  const pipe = await pipeline('automatic-speech-recognition', STT_MODEL_ID, {
    dtype,
    device,
    progress_callback: onProgress as never,
  });

  onProgress?.({ status: `ready on ${device}`, progress: 100 });
  return { pipe: pipe as unknown as WhisperPipeline, device };
}

export async function transcribe(loaded: LoadedSttModel, audio: Float32Array, _sampleRate: number): Promise<string> {
  let maxAmp = 0;
  for (let i = 0; i < audio.length; i++) {
    const abs = Math.abs(audio[i]!);
    if (abs > maxAmp) maxAmp = abs;
  }
  if (maxAmp < 0.005) return '';

  const out = await loaded.pipe(audio, {
    max_new_tokens: 64,
  });
  const result = Array.isArray(out) ? out[0] : out;
  if (!result || typeof result !== 'object' || !('text' in result)) return '';
  const raw = (result as { text: string }).text.trim();
  return cleanRepeatedTokens(raw);
}
