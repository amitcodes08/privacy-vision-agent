import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@huggingface/transformers', () => ({
  AutomaticSpeechRecognitionPipeline: class {},
  pipeline: vi.fn(),
  env: {
    allowLocalModels: true,
    useBrowserCache: false,
    backends: { onnx: { wasm: {} } },
  },
}));

vi.mock('~/ai/model-loader', () => ({
  probeWebGpu: vi.fn(),
}));

vi.mock('~/ai/ort-assets', () => ({
  ortWasmConfig: () => ({ wasmPaths: { wasm: '/ort/ort-wasm-simd-threaded.jsep.wasm' }, numThreads: 1 }),
}));

import { pipeline } from '@huggingface/transformers';
import { probeWebGpu } from '~/ai/model-loader';
import { STT_MODEL_ID, STT_MODEL_LABEL, STT_SIZE_MB, loadSttModel, transcribe } from '~/ai/stt-loader';

const mockPipelineFn = vi.fn();
const mockedPipeline = () => vi.mocked(pipeline) as unknown as { mockResolvedValue: (v: unknown) => void; mockRejectedValue: (e: unknown) => void };

function makeFakeProbe(available: boolean) {
  return vi.mocked(probeWebGpu).mockResolvedValue({
    available,
    adapter: available ? 'mock-adapter' : undefined,
    reason: available ? undefined : 'no WebGPU adapter',
    fp16: available,
  });
}

afterEach(() => {
  vi.clearAllMocks();
});

describe('STT_MODEL_ID', () => {
  it('is not a CDN URL', () => {
    expect(STT_MODEL_ID).not.toMatch(/cdn|jsdelivr|unpkg/i);
  });

  it('points at the onnx-community whisper-tiny.en repo', () => {
    expect(STT_MODEL_ID).toBe('onnx-community/whisper-tiny.en');
  });

  it('has a label and a size', () => {
    expect(typeof STT_MODEL_LABEL).toBe('string');
    expect(STT_MODEL_LABEL.length).toBeGreaterThan(0);
    expect(STT_SIZE_MB).toBeGreaterThan(0);
  });
});

describe('loadSttModel', () => {
  beforeEach(() => {
    mockedPipeline().mockResolvedValue(mockPipelineFn);
  });

  it('uses webgpu device when WebGPU is available', async () => {
    makeFakeProbe(true);
    const loaded = await loadSttModel();
    expect(loaded.device).toBe('webgpu');
    expect(vi.mocked(pipeline)).toHaveBeenCalledWith(
      'automatic-speech-recognition',
      STT_MODEL_ID,
      expect.objectContaining({ device: 'webgpu' }),
    );
  });

  it('falls back to wasm when WebGPU is unavailable', async () => {
    makeFakeProbe(false);
    const loaded = await loadSttModel();
    expect(loaded.device).toBe('wasm');
    expect(vi.mocked(pipeline)).toHaveBeenCalledWith(
      'automatic-speech-recognition',
      STT_MODEL_ID,
      expect.objectContaining({ device: 'wasm' }),
    );
  });

  it('uses q8 dtype on wasm', async () => {
    makeFakeProbe(false);
    await loadSttModel();
    expect(vi.mocked(pipeline)).toHaveBeenCalledWith(
      'automatic-speech-recognition',
      STT_MODEL_ID,
      expect.objectContaining({ dtype: 'q8' }),
    );
  });

  it('uses fp32 encoder + q4 decoder on webgpu', async () => {
    makeFakeProbe(true);
    await loadSttModel();
    expect(vi.mocked(pipeline)).toHaveBeenCalledWith(
      'automatic-speech-recognition',
      STT_MODEL_ID,
      expect.objectContaining({ dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } }),
    );
  });

  it('calls onProgress during load', async () => {
    makeFakeProbe(false);
    const onProgress = vi.fn();
    await loadSttModel(onProgress);
    expect(onProgress).toHaveBeenCalledWith(expect.objectContaining({ status: expect.stringContaining('loading') }));
    expect(onProgress).toHaveBeenCalledWith(expect.objectContaining({ status: expect.stringContaining('ready'), progress: 100 }));
  });

  it('returns a LoadedSttModel with a pipe property', async () => {
    makeFakeProbe(false);
    const loaded = await loadSttModel();
    expect(loaded.pipe).toBe(mockPipelineFn);
  });

  it('throws when pipeline() rejects', async () => {
    makeFakeProbe(false);
    mockedPipeline().mockRejectedValue(new Error('ort load failed'));
    await expect(loadSttModel()).rejects.toThrow('ort load failed');
  });
});

describe('transcribe', () => {
  const makeAudio = () => {
    const a = new Float32Array(160);
    a[0] = 0.5;
    return a;
  };

  it('returns the trimmed text from the pipeline output', async () => {
    const fakePipe = vi.fn().mockResolvedValue({ text: '  hello world  ' });
    const loaded = { pipe: fakePipe, device: 'wasm' as const } as unknown as import('~/ai/stt-loader').LoadedSttModel;
    const result = await transcribe(loaded, makeAudio(), 16_000);
    expect(result).toBe('hello world');
  });

  it('returns empty string for empty pipeline output', async () => {
    const fakePipe = vi.fn().mockResolvedValue({ text: '' });
    const loaded = { pipe: fakePipe, device: 'wasm' as const } as unknown as import('~/ai/stt-loader').LoadedSttModel;
    expect(await transcribe(loaded, makeAudio(), 16_000)).toBe('');
  });

  it('handles array output (pipeline may return array)', async () => {
    const fakePipe = vi.fn().mockResolvedValue([{ text: 'dictated text' }]);
    const loaded = { pipe: fakePipe, device: 'wasm' as const } as unknown as import('~/ai/stt-loader').LoadedSttModel;
    expect(await transcribe(loaded, makeAudio(), 16_000)).toBe('dictated text');
  });

  it('returns empty string when output has no text property', async () => {
    const fakePipe = vi.fn().mockResolvedValue({});
    const loaded = { pipe: fakePipe, device: 'wasm' as const } as unknown as import('~/ai/stt-loader').LoadedSttModel;
    expect(await transcribe(loaded, makeAudio(), 16_000)).toBe('');
  });

  it('returns empty string for silent audio without calling pipe', async () => {
    const fakePipe = vi.fn().mockResolvedValue({ text: 'hallucination' });
    const loaded = { pipe: fakePipe, device: 'wasm' as const } as unknown as import('~/ai/stt-loader').LoadedSttModel;
    const silentAudio = new Float32Array(160);
    expect(await transcribe(loaded, silentAudio, 16_000)).toBe('');
    expect(fakePipe).not.toHaveBeenCalled();
  });

  it('passes audio Float32Array and max_new_tokens to the pipeline', async () => {
    const fakePipe = vi.fn().mockResolvedValue({ text: 'ok' });
    const loaded = { pipe: fakePipe, device: 'wasm' as const } as unknown as import('~/ai/stt-loader').LoadedSttModel;
    await transcribe(loaded, makeAudio(), 16_000);
    expect(fakePipe).toHaveBeenCalledWith(
      expect.any(Float32Array),
      expect.objectContaining({ max_new_tokens: 64 }),
    );
  });
});
