import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SttWorkerRequest, SttWorkerResponse } from '~/ai/stt-worker';

vi.mock('~/ai/stt-loader', () => ({
  loadSttModel: vi.fn(),
  transcribe: vi.fn(),
}));

import { loadSttModel, transcribe } from '~/ai/stt-loader';

type HandlerFn = (ev: MessageEvent<SttWorkerRequest>) => void;

function makeWorkerEnv() {
  const posted: SttWorkerResponse[] = [];
  const listeners: HandlerFn[] = [];

  const selfMock = {
    postMessage: (msg: SttWorkerResponse) => posted.push(msg),
    addEventListener: (_: string, fn: HandlerFn) => listeners.push(fn),
  };
  vi.stubGlobal('self', selfMock);

  const dispatch = (req: SttWorkerRequest) =>
    listeners.forEach((fn) => fn({ data: req } as MessageEvent<SttWorkerRequest>));

  return { posted, dispatch };
}

async function flushPromises() {
  await new Promise((r) => setTimeout(r, 0));
}

afterEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe('stt-worker protocol', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('STT_PROBE before init responds with STT_READY device=none', async () => {
    const { posted, dispatch } = makeWorkerEnv();
    await import('~/ai/stt-worker');
    dispatch({ type: 'STT_PROBE', id: 'p1' });
    await flushPromises();
    expect(posted).toContainEqual(expect.objectContaining({ type: 'STT_READY', id: 'p1', device: 'none' }));
  });

  it('STT_INIT → STT_READY on success', async () => {
    const fakePipe = vi.fn().mockResolvedValue({ text: '' });
    vi.mocked(loadSttModel).mockResolvedValue({ pipe: fakePipe, device: 'wasm' } as unknown as import('~/ai/stt-loader').LoadedSttModel);
    const { posted, dispatch } = makeWorkerEnv();
    await import('~/ai/stt-worker');
    dispatch({ type: 'STT_INIT', id: 'i1' });
    await flushPromises();
    await flushPromises();
    expect(posted).toContainEqual(expect.objectContaining({ type: 'STT_READY', id: 'i1', device: 'wasm' }));
  });

  it('STT_INIT → ERROR when loadSttModel throws', async () => {
    vi.mocked(loadSttModel).mockRejectedValue(new Error('ort boom'));
    const { posted, dispatch } = makeWorkerEnv();
    await import('~/ai/stt-worker');
    dispatch({ type: 'STT_INIT', id: 'i2' });
    await flushPromises();
    await flushPromises();
    const err = posted.find((m) => m.type === 'ERROR' && m.id === 'i2');
    expect(err).toBeDefined();
    expect((err as Extract<SttWorkerResponse, { type: 'ERROR' }>).message).toContain('ort boom');
  });

  it('STT_TRANSCRIBE before init → ERROR with descriptive message', async () => {
    vi.mocked(loadSttModel).mockReturnValue(new Promise(() => {}));
    const { posted, dispatch } = makeWorkerEnv();
    await import('~/ai/stt-worker');
    dispatch({ type: 'STT_TRANSCRIBE', id: 't0', audio: new Float32Array(160), sampleRate: 16_000 });
    await flushPromises();
    await flushPromises();
    const err = posted.find((m) => m.type === 'ERROR' && m.id === 't0');
    expect(err).toBeDefined();
    expect((err as Extract<SttWorkerResponse, { type: 'ERROR' }>).message).toMatch(/not initialised/i);
  });

  it('STT_TRANSCRIBE after init → STT_RESULT with transcript', async () => {
    const fakePipe = vi.fn().mockResolvedValue({ text: 'search for shoes' });
    vi.mocked(loadSttModel).mockResolvedValue({ pipe: fakePipe, device: 'wasm' } as unknown as import('~/ai/stt-loader').LoadedSttModel);
    vi.mocked(transcribe).mockResolvedValue('search for shoes');
    const { posted, dispatch } = makeWorkerEnv();
    await import('~/ai/stt-worker');
    dispatch({ type: 'STT_INIT', id: 'i3' });
    await flushPromises();
    await flushPromises();
    dispatch({ type: 'STT_TRANSCRIBE', id: 't1', audio: new Float32Array(160), sampleRate: 16_000 });
    await flushPromises();
    await flushPromises();
    expect(posted).toContainEqual(expect.objectContaining({ type: 'STT_RESULT', id: 't1', transcript: 'search for shoes' }));
  });

  it('STT_TRANSCRIBE → ERROR when transcribe throws', async () => {
    const fakePipe = vi.fn();
    vi.mocked(loadSttModel).mockResolvedValue({ pipe: fakePipe, device: 'wasm' } as unknown as import('~/ai/stt-loader').LoadedSttModel);
    vi.mocked(transcribe).mockRejectedValue(new Error('decode failed'));
    const { posted, dispatch } = makeWorkerEnv();
    await import('~/ai/stt-worker');
    dispatch({ type: 'STT_INIT', id: 'i4' });
    await flushPromises();
    await flushPromises();
    dispatch({ type: 'STT_TRANSCRIBE', id: 't2', audio: new Float32Array(160), sampleRate: 16_000 });
    await flushPromises();
    await flushPromises();
    const err = posted.find((m) => m.type === 'ERROR' && m.id === 't2');
    expect(err).toBeDefined();
    expect((err as Extract<SttWorkerResponse, { type: 'ERROR' }>).message).toContain('decode failed');
  });

  it('STT_DISPOSE resets the model', async () => {
    const fakePipe = vi.fn().mockResolvedValue({ text: '' });
    vi.mocked(loadSttModel).mockResolvedValue({ pipe: fakePipe, device: 'wasm' } as unknown as import('~/ai/stt-loader').LoadedSttModel);
    vi.mocked(transcribe).mockResolvedValue('');
    const { posted, dispatch } = makeWorkerEnv();
    await import('~/ai/stt-worker');
    dispatch({ type: 'STT_INIT', id: 'i5' });
    await flushPromises();
    await flushPromises();
    dispatch({ type: 'STT_DISPOSE', id: 'd1' });
    await flushPromises();
    dispatch({ type: 'STT_TRANSCRIBE', id: 't3', audio: new Float32Array(160), sampleRate: 16_000 });
    await flushPromises();
    await flushPromises();
    const err = posted.find((m) => m.type === 'ERROR' && m.id === 't3');
    expect(err).toBeDefined();
    expect((err as Extract<SttWorkerResponse, { type: 'ERROR' }>).message).toMatch(/not initialised/i);
  });

  it('concurrent STT_INIT calls share a single loadSttModel call', async () => {
    const fakePipe = vi.fn();
    let resolve!: (v: import('~/ai/stt-loader').LoadedSttModel) => void;
    const pending = new Promise<import('~/ai/stt-loader').LoadedSttModel>((r) => { resolve = r; });
    vi.mocked(loadSttModel).mockReturnValue(pending);
    const { dispatch } = makeWorkerEnv();
    await import('~/ai/stt-worker');
    dispatch({ type: 'STT_INIT', id: 'ia' });
    dispatch({ type: 'STT_INIT', id: 'ib' });
    resolve({ pipe: fakePipe, device: 'wasm' } as unknown as import('~/ai/stt-loader').LoadedSttModel);
    await flushPromises();
    await flushPromises();
    expect(vi.mocked(loadSttModel)).toHaveBeenCalledTimes(1);
  });
});
