import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronDown,
  Cpu,
  ListChecks,
  Mic,
  MicOff,
  Play,
  Server,
  Settings2,
  ShieldCheck,
  Square,
  Terminal,
  Trash2,
} from 'lucide-react';
import type { AgentLogEntry, AgentStatus } from '@shared/types';
import { loadSettings, saveSettings, type Settings } from '~/lib/settings';

interface StatusReply {
  ok: boolean;
  status?: AgentStatus;
  logs?: AgentLogEntry[];
  error?: string;
}

interface PromptHistoryItem {
  id: string;
  prompt: string;
  createdAt: number;
}

const PROMPT_HISTORY_KEY = 'pva.promptHistory';
const MAX_PROMPT_HISTORY = 50;

async function loadPromptHistory(): Promise<PromptHistoryItem[]> {
  const raw = await chrome.storage.local.get(PROMPT_HISTORY_KEY);
  const value = raw[PROMPT_HISTORY_KEY];

  if (!Array.isArray(value)) return [];

  return value
    .filter(
      (item): item is PromptHistoryItem =>
        typeof item?.id === 'string' &&
        typeof item?.prompt === 'string' &&
        typeof item?.createdAt === 'number'
    )
    .sort((a, b) => b.createdAt - a.createdAt);
}

async function savePromptToHistory(prompt: string): Promise<PromptHistoryItem[]> {
  const trimmed = prompt.trim();
  if (!trimmed) return loadPromptHistory();

  const history = await loadPromptHistory();

  // Avoid storing the same prompt twice in a row.
  const withoutDuplicate = history.filter((item) => item.prompt !== trimmed);

  const next: PromptHistoryItem[] = [
    {
      id: crypto.randomUUID(),
      prompt: trimmed,
      createdAt: Date.now(),
    },
    ...withoutDuplicate,
  ].slice(0, MAX_PROMPT_HISTORY);

  await chrome.storage.local.set({
    [PROMPT_HISTORY_KEY]: next,
  });

  return next;
}

async function deletePromptFromHistory(id: string): Promise<PromptHistoryItem[]> {
  const history = await loadPromptHistory();
  const next = history.filter((item) => item.id !== id);

  await chrome.storage.local.set({
    [PROMPT_HISTORY_KEY]: next,
  });

  return next;
}

async function clearPromptHistory(): Promise<void> {
  await chrome.storage.local.remove(PROMPT_HISTORY_KEY);
}

type MicState = 'idle' | 'loading' | 'recording' | 'transcribing' | 'error';

const send = <T,>(msg: Record<string, unknown>) => chrome.runtime.sendMessage(msg) as Promise<T>;

function useStt(onTranscript: (t: string) => void) {
  const [micState, setMicState] = useState<MicState>('idle');
  const [micError, setMicError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    const cleanup = () => {
      recorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
    window.addEventListener('unload', cleanup);
    return () => window.removeEventListener('unload', cleanup);
  }, []);

  useEffect(() => {
    const handleFocus = () => {
      if (navigator.permissions?.query) {
        navigator.permissions
          .query({ name: 'microphone' as PermissionName })
          .then((res) => {
            if (res.state === 'granted') {
              setMicError(null);
            }
          })
          .catch(() => null);
      }
    };
    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, []);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === 'inactive') return;
    recorder.stop();
  }, []);

  const startRecording = useCallback(async (sttReady: boolean) => {
    setMicError(null);
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      if (err instanceof Error && err.name === 'NotAllowedError') {
        try {
          await chrome.tabs.create({ url: chrome.runtime.getURL('src/permission/index.html') });
          setMicError('Opening permission tab… Please allow microphone access, then click the mic again.');
        } catch {
          setMicError('Microphone permission required');
        }
      } else {
        setMicError('Microphone unavailable or blocked by system');
      }
      setMicState('idle');
      return;
    }

    if (!sttReady) {
      setMicState('loading');
      await send<{ ok: boolean }>({ kind: 'STT_WARM_UP' }).catch(() => null);
    }

    streamRef.current = stream;
    chunksRef.current = [];
    const mimeType =
      typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : '';
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    recorderRef.current = recorder;
    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setMicState('transcribing');
      try {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
        const arrayBuffer = await blob.arrayBuffer();
        const audioCtx = new AudioContext();
        if (audioCtx.state === 'suspended') {
          await audioCtx.resume();
        }
        const decoded = await audioCtx.decodeAudioData(arrayBuffer);
        await audioCtx.close();

        const TARGET_SR = 16_000;
        const offlineCtx = new OfflineAudioContext(1, Math.ceil(decoded.duration * TARGET_SR), TARGET_SR);
        const src = offlineCtx.createBufferSource();
        src.buffer = decoded;
        src.connect(offlineCtx.destination);
        src.start(0);
        const resampled = await offlineCtx.startRendering();
        const pcm = resampled.getChannelData(0);

        let maxAmp = 0;
        for (let i = 0; i < pcm.length; i++) {
          const abs = Math.abs(pcm[i]!);
          if (abs > maxAmp) maxAmp = abs;
        }
        if (maxAmp > 0 && maxAmp < 0.8) {
          const scale = 0.8 / maxAmp;
          for (let i = 0; i < pcm.length; i++) {
            pcm[i] = (pcm[i] ?? 0) * scale;
          }
        }

        const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!);
        const audioBase64 = btoa(binary);

        const r = await send<{ ok: boolean; transcript?: string; error?: string }>({
          kind: 'STT_TRANSCRIBE',
          audioBase64,
          sampleRate: TARGET_SR,
        });
        if (r.ok && r.transcript) onTranscript(r.transcript);
        else if (!r.ok) setMicError(r.error ?? 'Transcription failed');
      } catch (err) {
        setMicError(err instanceof Error ? err.message : 'Transcription failed');
      } finally {
        setMicState('idle');
      }
    };
    recorder.start();
    setMicState('recording');
  }, [onTranscript]);

  return { micState, micError, startRecording, stopRecording };
}

/** One short line for a log entry's `data`, whatever shape it arrived in. */
const detail = (d: unknown): string => {
  const s = typeof d === 'string' ? d : JSON.stringify(d);
  return s.length > 160 ? `${s.slice(0, 160)}…` : s;
};

export default function App() {
  const [goal, setGoal] = useState('');
  const [promptHistory, setPromptHistory] = useState<PromptHistoryItem[]>([]);
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [logs, setLogs] = useState<AgentLogEntry[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [openPanel, setOpenPanel] = useState<'none' | 'settings' | 'activity'>('none');

  const { micState, micError, startRecording, stopRecording } = useStt((t) => setGoal(t));

  const refresh = useCallback(async () => {
    const reply = await send<StatusReply>({ kind: 'AGENT_STATUS_REQUEST' }).catch(() => null);
    if (reply?.status) setStatus(reply.status);
    if (reply?.logs) setLogs(reply.logs);
  }, []);

  useEffect(() => {
    void loadSettings().then(setSettings);
    void loadPromptHistory().then(setPromptHistory);
    void refresh();

    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  }, [refresh]);

  const patch = async (p: Partial<Settings>) => setSettings(await saveSettings(p));

  const removePrompt = async (id: string) => {
    setPromptHistory(await deletePromptFromHistory(id));
  };

  const clearHistory = async () => {
    await clearPromptHistory();
    setPromptHistory([]);
  };

  const running = status?.running ?? false;

  const start = async () => {
    setErrorMsg(null);
    if (!goal.trim()) {
      setErrorMsg('Type what the agent should do on this tab.');
      return;
    }
    const prompt = goal.trim();
    setPromptHistory(await savePromptToHistory(prompt));
    setBusy(true);
    try {
      const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const res = await send<{ ok: boolean; error?: string }>({
        kind: 'AGENT_START',
        goal: prompt,
        tabId: activeTab?.id,
      });
      if (!res?.ok && res?.error) setErrorMsg(res.error);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      void refresh();
    }
  };

  /** One line that says where decisions are coming from. */
  const engine = useMemo(() => {
    if (!status) return { label: 'starting…', tone: 'idle' as const };
    if (status.modelLoading) {
      const pct = status.modelProgress;
      return { label: pct !== undefined ? `loading model ${pct}%` : (status.modelStage ?? 'loading model'), tone: 'busy' as const };
    }
    if (status.localModelReady) {
      return { label: status.webgpuAvailable ? 'on-device · WebGPU' : 'on-device · CPU', tone: 'ok' as const };
    }
    return { label: 'on-device planner', tone: 'warn' as const };
  }, [status]);

  /**
   * How the goal got sub-queried. Worth its own line: a one-step plan from
   * clause splitting and a one-step plan from Gemini Nano look identical in the
   * step counter but mean very different things when a run goes wrong.
   */
  const planner = useMemo(() => {
    const nano = status?.nano;
    if (!nano || nano.route === 'none') {
      return { label: 'rule-based sub-query', tone: 'warn' as const, title: nano?.reason ?? 'Chrome built-in AI not available here' };
    }
    if (nano.state !== 'available') {
      return { label: `Gemini Nano ${nano.state}`, tone: 'busy' as const, title: nano.reason ?? 'model not ready yet' };
    }
    return { label: 'Gemini Nano sub-query', tone: 'ok' as const, title: `Prompt API on the ${nano.route} context` };
  }, [status]);

  const plan = status?.plan;

  const problem = micError ?? errorMsg ?? status?.lastError ?? status?.modelError;
  const acted = (status?.localDecisions ?? 0) + (status?.heuristicDecisions ?? 0);
  const escalated = status?.escalations ?? 0;
  const hasRun = acted + escalated > 0;

  return (
    <div className="wrap">
      <header>
        <ShieldCheck size={15} className="brand" />
        <h1>Privacy Vision Agent</h1>
      </header>

      <div className="field">
        <textarea
          id="goal"
          value={goal}
          placeholder="What should the agent do on this tab?"
          rows={2}
          onChange={(e) => setGoal(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void start();
          }}
        />
        <button
          id="mic-btn"
          className="mic ghost"
          data-state={micState}
          disabled={micState === 'transcribing' || micState === 'loading'}
          title={
            micError ? micError :
            micState === 'loading' ? 'Loading STT model…' :
            micState === 'recording' ? 'Click to stop recording' :
            micState === 'transcribing' ? 'Transcribing…' :
            'Dictate your goal'
          }
          onClick={() => {
            if (micState === 'recording') stopRecording();
            else if (micState === 'idle' || micState === 'error') void startRecording(status?.sttReady ?? false);
          }}
        >
          {micState === 'recording' ? <MicOff size={14} /> : <Mic size={14} />}
        </button>
        <button
          className="primary"
          onClick={running ? () => void send({ kind: 'AGENT_STOP' }) : start}
          disabled={busy}
        >
          {running ? <Square size={13} /> : <Play size={13} />}
          {running ? `Stop · ${status?.step ?? 0}/${status?.maxSteps ?? '?'}` : 'Run'}
        </button>
      </div>

      {promptHistory.length > 0 && (
        <div className="prompt-history">
          <div className="prompt-history-header">
            <span className="prompt-history-title">Prompt history</span>

            <button
              className="prompt-history-clear"
              onClick={() => void clearHistory()}
              title="Delete all saved prompts"
            >
              Clear all
            </button>
          </div>

          <div className="prompt-history-list">
            {promptHistory.map((item) => (
              <div className="prompt-history-item" key={item.id}>
                <button
                  className="prompt-history-load"
                  title={item.prompt}
                  onClick={() => setGoal(item.prompt)}
                >
                  {item.prompt}
                </button>

                <button
                  className="prompt-history-delete"
                  title="Delete prompt"
                  aria-label={`Delete prompt: ${item.prompt}`}
                  onClick={() => void removePrompt(item.id)}
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="engine" data-tone={engine.tone}>
        <Cpu size={12} />
        <span>{engine.label}</span>
        {status?.modelLoading && status.modelProgress !== undefined && (
          <i className="bar" style={{ ['--p' as string]: `${status.modelProgress}%` }} />
        )}
        {hasRun && (
          <span className="tally" title="Actions decided on-device vs escalated to the server">
            {acted} local
            {escalated > 0 && (
              <>
                {' · '}
                <em>
                  <Server size={10} /> {escalated}
                </em>
              </>
            )}
          </span>
        )}
      </div>

      {(status?.sttLoading || micState === 'loading' || micState === 'transcribing') && (
        <div className="engine" data-tone="busy">
          <Mic size={12} />
          <span>
            {micState === 'transcribing' ? 'transcribing…' : (status?.sttStage ?? 'loading Whisper tiny')}
          </span>
          {status?.sttLoading && status.sttProgress !== undefined && (
            <i className="bar" style={{ ['--p' as string]: `${status.sttProgress}%` }} />
          )}
        </div>
      )}

      <div className="engine" data-tone={planner.tone} title={planner.title}>
        <ListChecks size={12} />
        <span>{planner.label}</span>
        {plan && plan.replans > 0 && (
          <span className="tally" title="Times the plan was rewritten against the live page">
            {plan.replans} re-plan{plan.replans > 1 ? 's' : ''}
          </span>
        )}
      </div>

      {plan && plan.objectives.length > 0 && (
        <ol className="plan">
          {plan.objectives.map((o, i) => (
            <li key={`${o.id ?? i}-${o.description}`} data-state={o.status}>
              <span className="tick">
                {o.status === 'completed' ? '✓' : o.status === 'skipped' ? '–' : o.status === 'active' ? '▸' : '·'}
              </span>
              {o.description}
            </li>
          ))}
        </ol>
      )}

      {problem && <div className="alert">{problem}</div>}

      <div className="disclosures">
        <button
          className={`disclosure ${openPanel === 'settings' ? 'open' : ''}`}
          onClick={() => setOpenPanel(openPanel === 'settings' ? 'none' : 'settings')}
        >
          <Settings2 size={12} /> Settings <ChevronDown size={12} className="chev" />
        </button>
        <button
          className={`disclosure ${openPanel === 'activity' ? 'open' : ''}`}
          onClick={() => setOpenPanel(openPanel === 'activity' ? 'none' : 'activity')}
        >
          <Terminal size={12} /> Activity <ChevronDown size={12} className="chev" />
        </button>
      </div>

      {openPanel === 'settings' && settings && (
        <div className="panel">
          <label className="check">
            <input
              type="checkbox"
              checked={settings.allowEscalation}
              onChange={(e) => void patch({ allowEscalation: e.target.checked })}
            />
            <span>
              Allow redacted cloud escalation
              <small>Only when nothing on-device is confident. Frames are redacted first.</small>
            </span>
          </label>

          <label className="check">
            <input
              type="checkbox"
              checked={settings.autoLoadModel}
              onChange={(e) => void patch({ autoLoadModel: e.target.checked })}
            />
            <span>
              Load the local model automatically
              <small>~230 MB on first run, then cached by the browser.</small>
            </span>
          </label>

          <div className="grid2">
            <div>
              <label htmlFor="thr">
                Confidence threshold <b>{settings.confidenceThreshold.toFixed(2)}</b>
              </label>
              <input
                id="thr"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={settings.confidenceThreshold}
                onChange={(e) => void patch({ confidenceThreshold: Number(e.target.value) })}
              />
            </div>
            <div>
              <label htmlFor="style">Redaction</label>
              <select
                id="style"
                value={settings.redactionStyle}
                onChange={(e) => void patch({ redactionStyle: e.target.value as Settings['redactionStyle'] })}
              >
                <option value="black">Black box</option>
                <option value="blur">Blur</option>
                <option value="pixelate">Pixelate</option>
              </select>
            </div>
          </div>

          <div>
            <label htmlFor="email">Profile email <small>stays on this device</small></label>
            <input
              id="email"
              type="email"
              placeholder="you@example.com"
              value={settings.profile.email ?? ''}
              onChange={(e) => void patch({ profile: { ...settings.profile, email: e.target.value } })}
            />
          </div>

          {!status?.localModelReady && !status?.modelLoading && (
            <button className="ghost wide" onClick={() => void send({ kind: 'WARM_UP' }).then(refresh)}>
              Load local model now
            </button>
          )}
          {status?.localModelId && <div className="meta">{status.localModelId}</div>}
        </div>
      )}

      {openPanel === 'activity' && (
        <div className="panel log">
          {logs.length === 0 && <div className="meta">No activity yet.</div>}
          {logs
            .slice()
            .reverse()
            .map((l, i) => (
              <div key={`${l.ts}-${i}`} className={l.level}>
                <span className="t">{new Date(l.ts).toLocaleTimeString()}</span>
                <span className="s">{l.scope}</span>
                {l.message}
                {/* The cause of a failure often lived only in `data`, which was
                    never rendered — so a load error read as an unexplained
                    "model init failed". */}
                {l.data !== undefined && l.data !== '' && <span className="d">{detail(l.data)}</span>}
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
