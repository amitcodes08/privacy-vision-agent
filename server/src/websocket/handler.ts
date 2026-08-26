/**
 * WebSocket handler: envelope validation, per-session rate limiting, and a
 * hard latency budget on cloud inference.
 */
import { WebSocketServer, WebSocket, type RawData } from 'ws';
import type { IncomingMessage, Server } from 'node:http';
import {
  DEFAULTS,
  envelope,
  isEnvelope,
  newId,
  type ClientMessage,
  type ErrorPayload,
  type InferenceRequestPayload,
  type WsErrorCode,
} from '../../../shared/types.ts';
import { activeModelId, planAction } from '../ai/cloud-vlm.ts';

const LATENCY_BUDGET_MS = Number(process.env.LATENCY_BUDGET_MS ?? 1500);
const MAX_REQUESTS_PER_MIN = Number(process.env.MAX_REQUESTS_PER_MIN ?? 60);

interface Session {
  id: string;
  clientId?: string;
  alive: boolean;
  windowStart: number;
  count: number;
}

const sessions = new WeakMap<WebSocket, Session>();

export interface HandlerStats {
  connections: number;
  requests: number;
  errors: number;
  p50LatencyMs: number;
}

const latencies: number[] = [];
const stats: HandlerStats = { connections: 0, requests: 0, errors: 0, p50LatencyMs: 0 };
export const getStats = (): HandlerStats => ({ ...stats, p50LatencyMs: percentile(latencies, 0.5) });

export function attachWebSocket(server: Server): WebSocketServer {
  const wss = new WebSocketServer({ server, maxPayload: DEFAULTS.maxPayloadBytes, path: '/' });

  wss.on('connection', (ws: WebSocket, req: IncomingMessage) => {
    const session: Session = { id: newId(), alive: true, windowStart: Date.now(), count: 0 };
    sessions.set(ws, session);
    stats.connections++;
    console.log(`[ws] + ${session.id} from ${req.socket.remoteAddress ?? '?'}`);

    ws.on('pong', () => {
      session.alive = true;
    });
    ws.on('message', (data) => void onMessage(ws, session, data));
    ws.on('close', (code) => console.log(`[ws] - ${session.id} (${code})`));
    ws.on('error', (err) => console.error(`[ws] ! ${session.id}`, err.message));
  });

  // Drop half-open sockets so a sleeping laptop cannot leak sessions.
  const sweep = setInterval(() => {
    for (const ws of wss.clients) {
      const s = sessions.get(ws);
      if (!s) continue;
      if (!s.alive) {
        ws.terminate();
        continue;
      }
      s.alive = false;
      ws.ping();
    }
  }, DEFAULTS.heartbeatIntervalMs);
  wss.on('close', () => clearInterval(sweep));

  return wss;
}

async function onMessage(ws: WebSocket, session: Session, data: RawData): Promise<void> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data.toString('utf8'));
  } catch {
    return fail(ws, 'BAD_ENVELOPE', 'frame is not JSON');
  }
  if (!isEnvelope(parsed)) return fail(ws, 'BAD_ENVELOPE', 'missing protocol fields');
  const msg = parsed as ClientMessage;

  switch (msg.type) {
    case 'CONNECT': {
      session.clientId = msg.payload.clientId;
      send(
        ws,
        envelope('CONNECT_ACK', {
          sessionId: session.id,
          serverModelId: activeModelId,
          heartbeatIntervalMs: DEFAULTS.heartbeatIntervalMs,
        }),
      );
      console.log(
        `[ws] ${session.id} client=${msg.payload.clientId} webgpu=${msg.payload.capabilities.webgpu} local=${msg.payload.capabilities.localModelId ?? 'none'}`,
      );
      return;
    }
    case 'PING':
      send(ws, envelope('PONG', {} as Record<string, never>, msg.id));
      return;
    case 'INFERENCE_REQUEST':
      return void handleInference(ws, session, msg.id, msg.payload);
    default:
      return fail(ws, 'BAD_ENVELOPE', `unsupported type ${(msg as { type: string }).type}`);
  }
}

async function handleInference(
  ws: WebSocket,
  session: Session,
  requestId: string,
  payload: InferenceRequestPayload,
): Promise<void> {
  if (rateLimited(session)) return fail(ws, 'RATE_LIMITED', `>${MAX_REQUESTS_PER_MIN} req/min`, requestId);
  const invalid = validate(payload);
  if (invalid) return fail(ws, 'BAD_ENVELOPE', invalid, requestId);

  stats.requests++;
  const started = Date.now();
  try {
    const result = await withTimeout(planAction(payload), LATENCY_BUDGET_MS * 4);
    const elapsed = Date.now() - started;
    latencies.push(elapsed);
    if (latencies.length > 500) latencies.shift();
    if (elapsed > LATENCY_BUDGET_MS) console.warn(`[ws] latency budget exceeded: ${elapsed}ms`);
    send(ws, envelope('INFERENCE_RESPONSE', result, requestId));
  } catch (err) {
    stats.errors++;
    fail(ws, 'MODEL_ERROR', err instanceof Error ? err.message : String(err), requestId);
  }
}

function validate(p: InferenceRequestPayload): string | null {
  if (typeof p?.goal !== 'string' || p.goal.length === 0) return 'goal is required';
  if (typeof p.imageBase64 !== 'string' || p.imageBase64.length < 64) return 'imageBase64 missing';
  if (!p.dom || !Array.isArray(p.dom.nodes)) return 'dom.nodes missing';
  if (p.dom.nodes.length > 1000) return 'dom.nodes too large';
  return null;
}

function rateLimited(s: Session): boolean {
  const now = Date.now();
  if (now - s.windowStart > 60_000) {
    s.windowStart = now;
    s.count = 0;
  }
  return ++s.count > MAX_REQUESTS_PER_MIN;
}

function send(ws: WebSocket, msg: unknown): void {
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function fail(ws: WebSocket, code: WsErrorCode, message: string, requestId?: string): void {
  stats.errors++;
  const payload: ErrorPayload = { code, message, requestId };
  send(ws, envelope('ERROR', payload));
  console.warn(`[ws] error ${code}: ${message}`);
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`inference exceeded ${ms}ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

function percentile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))] ?? 0;
}
