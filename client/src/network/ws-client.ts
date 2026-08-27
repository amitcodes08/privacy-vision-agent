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

/**
 * Give up after this many consecutive failures and go dormant. The socket only
 * exists to serve `infer()`, so retrying forever on a timer just burns a socket
 * every backoff interval — the reconnect ladder that reached "#11 in 36093ms"
 * with the server simply not running. A later `infer()` re-arms it.
 */
const MAX_ATTEMPTS = 5;

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

  /**
   * Idempotent. Every guard here matters:
   *
   * The old condition was `this.ws && (state === 'open' || state === 'connecting')`,
   * which let a call through whenever a reconnect was merely *pending* — and
   * `background/index.ts` called `connect()` on every escalating step. Each call
   * built a second WebSocket while the first was still live and its
   * reconnect timer still armed. The abandoned socket kept its listeners, so its
   * eventual `close` ran `handleClose` too and scheduled its own reconnect. That
   * is the duplicated `reconnect #7` / `#8` inside the same second, with the
   * attempt counter and the backoff both running away.
   */
  connect(): void {
    this.stopped = false;
    if (this.reconnectTimer) return; // a retry is already queued
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) return;
    if (this.attempt >= MAX_ATTEMPTS) {
      this.log(`dormant after ${this.attempt} failed attempts; a new request will retry`);
      this.setState('closed');
      return;
    }
    this.setState(this.attempt === 0 ? 'connecting' : 'reconnecting');

    let socket: WebSocket;
    try {
      socket = new WebSocket(this.url);
    } catch (err) {
      this.log('constructor threw', err);
      this.scheduleReconnect();
      return;
    }
    this.ws = socket;
    socket.addEventListener('open', () => this.handleOpen(socket));
    socket.addEventListener('message', this.handleMessage);
    socket.addEventListener('close', (ev) => this.handleClose(socket, ev));
    // Only the current socket may speak for the client. Without this a
    // superseded socket's close event resets `this.ws` to null underneath a
    // healthy connection.
    socket.addEventListener('error', () => {
      if (socket === this.ws) this.log('socket error');
    });
  }

  close(): void {
    this.stopped = true;
    this.attempt = 0;
    this.clearTimers();
    this.rejectAll(new Error('client closed'));
    const socket = this.ws;
    this.ws = null;
    socket?.close(1000, 'client shutdown');
    this.setState('closed');
  }

  /** Escalate one frame. Rejects on timeout, server error, or disconnect. */
  async infer(payload: InferenceRequestPayload, timeoutMs?: number): Promise<InferenceResponsePayload> {
    const msg = envelope('INFERENCE_REQUEST', payload);
    const raw = JSON.stringify(msg);
    if (raw.length > DEFAULTS.maxPayloadBytes) {
      throw new Error(`payload ${raw.length}B exceeds cap ${DEFAULTS.maxPayloadBytes}B`);
    }
    if (!this.connected) {
      // Real demand, so forgive an exhausted attempt budget and try again now
      // rather than waiting out a backoff nobody is watching.
      if (this.attempt >= MAX_ATTEMPTS || this.reconnectTimer) {
        this.clearTimers();
        this.attempt = 0;
      }
      this.connect();
      await this.waitForOpen(3_000);
    }
    return new Promise<InferenceResponsePayload>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(msg.id);
        reject(new Error('escalation timed out (model generation took >60s)'));
      }, timeoutMs ?? this.opts.requestTimeoutMs ?? 60_000);
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

  private handleOpen = (socket: WebSocket): void => {
    if (socket !== this.ws) return socket.close(1000, 'superseded');
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

  private handleClose = (socket: WebSocket, ev: CloseEvent): void => {
    // A socket we already replaced must not touch shared state — otherwise its
    // close nulls `this.ws`, rejects the live connection's pending requests, and
    // queues a reconnect that races the real one.
    if (socket !== this.ws) return;
    this.clearTimers();
    this.rejectAll(new Error(`socket closed (${ev.code})`));
    this.ws = null;
    if (this.stopped) return this.setState('closed');
    this.scheduleReconnect();
  };

  private scheduleReconnect(): void {
    this.attempt++;
    if (this.attempt >= MAX_ATTEMPTS) {
      this.log(`giving up after ${this.attempt} attempts; dormant until the next request`);
      this.setState('closed');
      return;
    }
    this.setState('reconnecting');
    // 8s ceiling rather than 30s: this is localhost, and the attempt cap keeps
    // the whole ladder short enough that a user watching the log can follow it.
    const backoff = Math.min(8_000, 500 * 2 ** (this.attempt - 1)) * (0.75 + Math.random() * 0.5);
    this.log(`reconnect #${this.attempt} in ${Math.round(backoff)}ms`);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, backoff);
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
