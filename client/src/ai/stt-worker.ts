/// <reference lib="webworker" />
import { loadSttModel, transcribe, type LoadedSttModel, type SttProgressFn } from './stt-loader';

export type SttWorkerRequest =
  | { type: 'STT_PROBE'; id: string }
  | { type: 'STT_INIT'; id: string }
  | { type: 'STT_TRANSCRIBE'; id: string; audio: Float32Array; sampleRate: number }
  | { type: 'STT_DISPOSE'; id: string };

export type SttWorkerResponse =
  | { type: 'STT_PROGRESS'; id: string; status: string; file?: string; progress?: number; loaded?: number; total?: number }
  | { type: 'STT_READY'; id: string; device: string }
  | { type: 'STT_RESULT'; id: string; transcript: string }
  | { type: 'ERROR'; id: string; message: string };

let loaded: LoadedSttModel | null = null;
let loading: Promise<LoadedSttModel> | null = null;

const post = (msg: SttWorkerResponse) => self.postMessage(msg);

self.addEventListener('message', (event: MessageEvent<SttWorkerRequest>) => {
  void handle(event.data);
});

async function handle(req: SttWorkerRequest): Promise<void> {
  try {
    switch (req.type) {
      case 'STT_PROBE': {
        post({ type: 'STT_READY', id: req.id, device: loaded?.device ?? 'none' });
        return;
      }
      case 'STT_INIT': {
        const onProgress: SttProgressFn = (p) =>
          post({
            type: 'STT_PROGRESS',
            id: req.id,
            status: p.status,
            file: p.file,
            progress: p.progress,
            loaded: p.loaded,
            total: p.total,
          });
        try {
          loading ??= loadSttModel(onProgress);
          loaded = await loading;
          post({ type: 'STT_READY', id: req.id, device: loaded.device });
        } catch (err) {
          loading = null;
          throw err;
        }
        return;
      }
      case 'STT_TRANSCRIBE': {
        if (!loaded && loading) {
          loaded = await loading.catch(() => null);
        }
        if (!loaded) {
          post({ type: 'ERROR', id: req.id, message: 'STT model not initialised' });
          return;
        }
        const transcript = await transcribe(loaded, req.audio, req.sampleRate);
        post({ type: 'STT_RESULT', id: req.id, transcript });
        return;
      }
      case 'STT_DISPOSE': {
        loaded = null;
        loading = null;
        return;
      }
    }
  } catch (err) {
    post({ type: 'ERROR', id: (req as { id: string }).id, message: err instanceof Error ? err.message : String(err) });
  }
}
