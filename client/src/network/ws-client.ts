/**
 * Auto-reconnecting WebSocket client with request/response correlation.
 * Lives in the service worker; every send is validated against the shared
 * envelope contract before it hits the wire.
 */
import {
  DEFAULTS,
  envelope,
  isEnvelope,
  newId,
  type ConnectAckPayload,
  type ErrorPayload,
  type InferenceRequestPayload,
  type InferenceResponsePayload,
  type ServerMessage,
} from '@shared/types';

export type WsState = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed';

export interface WsClientOptions {
  url?: string;
  clientId?: string;
  extensionVersion?: string;
  webgpu?: boolean;
  localModelId?: string;
  requestTimeoutMs?: number;
  onState?: (state: WsState) => void;
  onError?: (err: ErrorPayload) => void;
  onLog?: (msg: string, data?: unknown) => void;
}

interface Pending {
  resolve: (p: InferenceResponsePayload) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class WsClient {
  private ws: WebSocket | null = null;
  private state: WsState = 'idle';
  private attempt = 0;
  private heartbeat: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly pending = new Map<string, Pending>();
  private ack: ConnectAckPayload | null = null;
  private stopped = false;
  private readonly clientId: string;

  constructor(private readonly opts: WsClientOptions = {}) {
    this.clientId = opts.clientId ?? newId();
  }

  get url(): string {
    return this.opts.url ?? DEFAULTS.wsUrl;
  }
  get connected(): boolean {
    return this.state === 'open';
  }
  get session(): ConnectAckPayload | null {
    return this.ack;
  }

  connect(): void {
    this.stopped = false;
    if (this.ws && (this.state === 'open' || this.state === 'connecting')) return;
    this.setState(this.attempt === 0 ? 'connecting' : 'reconnecting');
    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      this.log('constructor threw', err);
      this.scheduleReconnect();
      return;
    }
    this.ws.addEventListener('open', this.handleOpen);
    this.ws.addEventListener('message', this.handleMessage);
    this.ws.addEventListener('close', this.handleClose);
    this.ws.addEventListener('error', () => this.log('socket error'));
  }

  close(): void {
    this.stopped = true;
    this.clearTimers();
    this.rejectAll(new Error('client closed'));
    this.ws?.close(1000, 'client shutdown');
    this.ws = null;
    this.setState('closed');
  }

  /** Escalate one frame. Rejects on timeout, server error, or disconnect. */
  async infer(payload: InferenceRequestPayload, timeoutMs?: number): Promise<InferenceResponsePayload> {
    const msg = envelope('INFERENCE_REQUEST', payload);
    const raw = JSON.stringify(msg);
    if (raw.length > DEFAULTS.maxPayloadBytes) {
      throw new Error(`payload ${raw.length}B exceeds cap ${DEFAULTS.maxPayloadBytes}B`);
    }
    if (!this.connected || !this.ws) {
      this.connect();
      await this.waitForOpen(3_000);
    }
    return new Promise<InferenceResponsePayload>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(msg.id);
        reject(new Error('escalation timed out'));
      }, timeoutMs ?? this.opts.requestTimeoutMs ?? 8_000);
      this.pending.set(msg.id, { resolve, reject, timer });
      this.ws!.send(raw);
    });
  }

  private waitForOpen(ms: number): Promise<void> {
    if (this.connected) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const started = Date.now();
      const tick = setInterval(() => {
        if (this.connected) {
          clearInterval(tick);
          resolve();
        } else if (Date.now() - started > ms) {
          clearInterval(tick);
          reject(new Error('websocket unavailable'));
        }
      }, 50);
    });
  }

  private handleOpen = (): void => {
    this.attempt = 0;
    this.setState('open');
    this.send(
      envelope('CONNECT', {
        clientId: this.clientId,
        extensionVersion: this.opts.extensionVersion ?? '0.0.0',
        capabilities: { webgpu: this.opts.webgpu ?? false, localModelId: this.opts.localModelId },
      }),
    );
    this.startHeartbeat(DEFAULTS.heartbeatIntervalMs);
  };

  private handleMessage = (ev: MessageEvent): void => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(typeof ev.data === 'string' ? ev.data : '');
    } catch {
      this.log('non-JSON frame dropped');
      return;
    }
    if (!isEnvelope(parsed)) {
      this.log('bad envelope dropped', parsed);
      return;
    }
    const msg = parsed as ServerMessage;
    switch (msg.type) {
      case 'CONNECT_ACK':
        this.ack = msg.payload;
        this.startHeartbeat(msg.payload.heartbeatIntervalMs);
        this.log('session established', msg.payload.sessionId);
        break;
      case 'INFERENCE_RESPONSE': {
        const p = this.pending.get(msg.id);
        if (!p) return this.log('unmatched response', msg.id);
        clearTimeout(p.timer);
        this.pending.delete(msg.id);
        p.resolve(msg.payload);
        break;
      }
      case 'ERROR': {
        const { requestId } = msg.payload;
        this.opts.onError?.(msg.payload);
        if (requestId) {
          const p = this.pending.get(requestId);
          if (p) {
            clearTimeout(p.timer);
            this.pending.delete(requestId);
            p.reject(new Error(`${msg.payload.code}: ${msg.payload.message}`));
          }
        }
        break;
      }
      case 'PONG':
        break;
    }
  };

  private handleClose = (ev: CloseEvent): void => {
    this.clearTimers();
    this.rejectAll(new Error(`socket closed (${ev.code})`));
    this.ws = null;
    if (this.stopped) return this.setState('closed');
    this.scheduleReconnect();
  };

  private scheduleReconnect(): void {
    this.setState('reconnecting');
    const backoff = Math.min(30_000, 500 * 2 ** this.attempt) * (0.75 + Math.random() * 0.5);
    this.attempt++;
    this.log(`reconnect #${this.attempt} in ${Math.round(backoff)}ms`);
    this.reconnectTimer = setTimeout(() => this.connect(), backoff);
  }

  private startHeartbeat(intervalMs: number): void {
    if (this.heartbeat) clearInterval(this.heartbeat);
    this.heartbeat = setInterval(() => {
      if (this.connected) this.send(envelope('PING', {} as Record<string, never>));
    }, Math.max(5_000, intervalMs));
  }

  private send(msg: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  private clearTimers(): void {
    if (this.heartbeat) clearInterval(this.heartbeat);
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.heartbeat = null;
    this.reconnectTimer = null;
  }

  private rejectAll(err: Error): void {
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(err);
    }
    this.pending.clear();
  }

  private setState(s: WsState): void {
    if (this.state === s) return;
    this.state = s;
    this.opts.onState?.(s);
  }

  private log(msg: string, data?: unknown): void {
    this.opts.onLog?.(`[ws] ${msg}`, data);
  }
}
