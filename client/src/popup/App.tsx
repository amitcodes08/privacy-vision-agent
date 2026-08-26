import { useCallback, useEffect, useState } from 'react';
import { Cpu, Eye, Play, Radio, Shield, Square } from 'lucide-react';
import type { AgentLogEntry, AgentStatus } from '@shared/types';
import { loadSettings, saveSettings, type Settings } from '~/lib/settings';

interface StatusReply {
  ok: boolean;
  status?: AgentStatus;
  logs?: AgentLogEntry[];
  error?: string;
}

const send = <T,>(msg: Record<string, unknown>) => chrome.runtime.sendMessage(msg) as Promise<T>;

export default function App() {
  const [goal, setGoal] = useState('');
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [logs, setLogs] = useState<AgentLogEntry[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    const reply = await send<StatusReply>({ kind: 'AGENT_STATUS_REQUEST' }).catch(() => null);
    if (reply?.status) setStatus(reply.status);
    if (reply?.logs) setLogs(reply.logs);
  }, []);

  useEffect(() => {
    void loadSettings().then(setSettings);
    void refresh();
    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  }, [refresh]);

  const patch = async (p: Partial<Settings>) => setSettings(await saveSettings(p));

  const start = async () => {
    if (!goal.trim()) return;
    setBusy(true);
    await send({ kind: 'AGENT_START', goal: goal.trim() }).catch(() => null);
    setBusy(false);
    void refresh();
  };

  const running = status?.running ?? false;

  return (
    <div className="wrap">
      <header>
        <Shield size={16} color="#7c3aed" />
        <h1>Privacy Vision Agent</h1>
        <span className="spacer" />
        <span className={`pill ${status?.webgpuAvailable ? 'on' : 'warn'}`} title="WebGPU adapter">
          <Cpu size={11} /> {status?.webgpuAvailable ? 'WebGPU' : 'no GPU'}
        </span>
        <span className={`pill ${status?.wsConnected ? 'on' : 'off'}`} title="Escalation channel">
          <Radio size={11} /> {status?.wsConnected ? 'ws' : 'local'}
        </span>
      </header>

      <div className="panel">
        <label htmlFor="goal">Goal for this tab</label>
        <textarea
          id="goal"
          value={goal}
          placeholder="e.g. accept cookies, then open the billing settings page"
          onChange={(e) => setGoal(e.target.value)}
        />
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={running ? () => void send({ kind: 'AGENT_STOP' }) : start} disabled={busy}>
            {running ? <Square size={13} /> : <Play size={13} />}
            {running ? 'Stop' : 'Run agent'}
          </button>
          <button className="ghost" onClick={() => void send({ kind: 'WARM_UP' }).then(refresh)}>
            <Eye size={13} /> {status?.localModelReady ? 'Model loaded' : 'Load local model'}
          </button>
        </div>
        {status?.localModelId && <div className="muted" style={{ marginTop: 6 }}>{status.localModelId}</div>}
      </div>

      <div className="stats">
        <div className="stat">
          <b>{status?.localDecisions ?? 0}</b>
          <span>local acts</span>
        </div>
        <div className="stat">
          <b>{status?.escalations ?? 0}</b>
          <span>escalations</span>
        </div>
        <div className="stat">
          <b>{status?.redactions ?? 0}</b>
          <span>redactions</span>
        </div>
      </div>

      {settings && (
        <div className="panel">
          <div className="row">
            <div style={{ flex: 1 }}>
              <label htmlFor="thr">Escalation threshold ({settings.confidenceThreshold.toFixed(2)})</label>
              <input
                id="thr"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={settings.confidenceThreshold}
                onChange={(e) => void patch({ confidenceThreshold: Number(e.target.value) })}
                style={{ width: '100%' }}
              />
            </div>
            <div style={{ width: 110 }}>
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
          <label style={{ marginTop: 8 }}>
            <input
              type="checkbox"
              checked={settings.allowEscalation}
              onChange={(e) => void patch({ allowEscalation: e.target.checked })}
            />{' '}
            Allow redacted cloud escalation
          </label>
          <label htmlFor="email" style={{ marginTop: 8 }}>Local profile — email (never sent to server)</label>
          <input
            id="email"
            type="email"
            value={settings.profile.email ?? ''}
            onChange={(e) => void patch({ profile: { ...settings.profile, email: e.target.value } })}
          />
        </div>
      )}

      <div className="panel log">
        {logs.length === 0 && <div>no activity yet</div>}
        {logs.map((l, i) => (
          <div key={`${l.ts}-${i}`} className={l.level}>
            {new Date(l.ts).toLocaleTimeString()} [{l.scope}] {l.message}
          </div>
        ))}
      </div>
    </div>
  );
}
