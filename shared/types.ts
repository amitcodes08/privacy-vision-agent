/**
 * Shared wire contracts between the Chrome extension (client) and the
 * escalation server. This file is the single source of truth for both
 * workspaces — keep it dependency-free so it can be imported from a
 * service worker, a web worker, and Node without bundler gymnastics.
 */

export const PROTOCOL_VERSION = 1 as const;

/* ------------------------------------------------------------------ *
 * Agent actions
 * ------------------------------------------------------------------ */

export type ActionKind =
  | 'click'
  | 'fill'
  | 'scroll'
  | 'navigate'
  | 'wait'
  | 'escalate'
  | 'done';

/**
 * Placeholders the *server* is allowed to emit instead of raw values.
 * The client hydrates these from local private context so secrets never
 * leave the browser. See `client/src/content/value-hydrator.ts`.
 */
export type ValueToken =
  | 'USER_EMAIL'
  | 'USER_FULL_NAME'
  | 'USER_PHONE'
  | 'USER_ADDRESS'
  | 'USER_PASSWORD'
  | 'OTP_CODE'
  | 'LITERAL';

export interface ClickAction {
  action: 'click';
  selector: string;
  reason?: string;
}

export interface FillAction {
  action: 'fill';
  selector: string;
  /** Where the client should source the text from. */
  valueType: ValueToken;
  /** Only populated when `valueType === 'LITERAL'` (non-sensitive text). */
  value?: string;
  submit?: boolean;
  reason?: string;
}

export interface ScrollAction {
  action: 'scroll';
  /** CSS selector to scroll into view; omit to scroll the viewport. */
  selector?: string;
  deltaY?: number;
  reason?: string;
}

export interface NavigateAction {
  action: 'navigate';
  url: string;
  reason?: string;
}

export interface WaitAction {
  action: 'wait';
  ms: number;
  reason?: string;
}

/** Emitted by the *local* model only: "I cannot decide, ask the cloud." */
export interface EscalateAction {
  action: 'escalate';
  reason?: string;
}

export interface DoneAction {
  action: 'done';
  summary?: string;
}

export type AgentAction =
  | ClickAction
  | FillAction
  | ScrollAction
  | NavigateAction
  | WaitAction
  | EscalateAction
  | DoneAction;

/** An action plus the metadata used by the escalation gate. */
export interface AgentDecision {
  action: AgentAction;
  /** 0..1 — below `confidenceThreshold` triggers escalation. */
  confidence: number;
  source: 'local' | 'cloud' | 'heuristic';
  latencyMs?: number;
  modelId?: string;
}

/* ------------------------------------------------------------------ *
 * Scrubbed DOM representation
 * ------------------------------------------------------------------ */

export type RedactionReason =
  | 'password'
  | 'credit-card'
  | 'email'
  | 'phone'
  | 'ssn'
  | 'aadhaar'
  | 'otp'
  | 'autocomplete-sensitive'
  | 'user-marked';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * A flattened, interaction-oriented view of the page. Text of sensitive
 * nodes is already replaced with `REDACTED_PLACEHOLDER` on the client.
 */
export interface ScrubbedNode {
  /** Stable index used by the model to reference the node. */
  id: number;
  tag: string;
  /** Deterministic selector the client can resolve back to a live node. */
  selector: string;
  role?: string;
  type?: string;
  name?: string;
  label?: string;
  placeholder?: string;
  text?: string;
  value?: string;
  href?: string;
  checked?: boolean;
  disabled?: boolean;
  visible: boolean;
  box?: BoundingBox;
  redacted?: RedactionReason[];
}

export interface ScrubbedDom {
  url: string;
  /** Origin only — never the full URL when it may embed tokens. */
  origin: string;
  title: string;
  viewport: { width: number; height: number; scrollX: number; scrollY: number };
  nodes: ScrubbedNode[];
  /** Counts by reason, for the popup's privacy receipt. */
  redactionSummary: Partial<Record<RedactionReason, number>>;
}

export const REDACTED_PLACEHOLDER = '[REDACTED]' as const;

/* ------------------------------------------------------------------ *
 * WebSocket envelope
 * ------------------------------------------------------------------ */

export type WsMessageType =
  | 'CONNECT'
  | 'CONNECT_ACK'
  | 'INFERENCE_REQUEST'
  | 'INFERENCE_RESPONSE'
  | 'PING'
  | 'PONG'
  | 'ERROR';

export interface WsEnvelope<T extends WsMessageType, P> {
  v: typeof PROTOCOL_VERSION;
  type: T;
  /** Correlates request/response pairs. */
  id: string;
  ts: number;
  payload: P;
}

export interface ConnectPayload {
  clientId: string;
  extensionVersion: string;
  capabilities: {
    webgpu: boolean;
    localModelId?: string;
  };
}

export interface ConnectAckPayload {
  sessionId: string;
  serverModelId: string;
  heartbeatIntervalMs: number;
}

export interface InferenceRequestPayload {
  /** Natural-language goal, e.g. "log in and open billing settings". */
  goal: string;
  /** Redacted JPEG frame, base64 *without* the data: prefix. */
  imageBase64: string;
  imageMime: 'image/jpeg' | 'image/webp';
  dom: ScrubbedDom;
  /** Prior actions this session, for context. */
  history?: AgentAction[];
  /** Why the local model gave up. */
  localConfidence?: number;
  localReason?: string;
}

export interface InferenceResponsePayload {
  decision: AgentDecision;
  /** Raw model text, retained for debugging in the popup. */
  rationale?: string;
  usage?: { inputTokens?: number; outputTokens?: number };
}

export type WsErrorCode =
  | 'BAD_ENVELOPE'
  | 'UNSUPPORTED_VERSION'
  | 'PAYLOAD_TOO_LARGE'
  | 'MODEL_ERROR'
  | 'RATE_LIMITED'
  | 'INTERNAL';

export interface ErrorPayload {
  code: WsErrorCode;
  message: string;
  /** Envelope id this error refers to, when known. */
  requestId?: string;
}

export type ClientMessage =
  | WsEnvelope<'CONNECT', ConnectPayload>
  | WsEnvelope<'INFERENCE_REQUEST', InferenceRequestPayload>
  | WsEnvelope<'PING', Record<string, never>>;

export type ServerMessage =
  | WsEnvelope<'CONNECT_ACK', ConnectAckPayload>
  | WsEnvelope<'INFERENCE_RESPONSE', InferenceResponsePayload>
  | WsEnvelope<'PONG', Record<string, never>>
  | WsEnvelope<'ERROR', ErrorPayload>;

export type AnyMessage = ClientMessage | ServerMessage;

/* ------------------------------------------------------------------ *
 * Runtime guards & helpers (no deps, usable on both sides)
 * ------------------------------------------------------------------ */

export const DEFAULTS = {
  wsUrl: 'ws://localhost:8080',
  confidenceThreshold: 0.62,
  heartbeatIntervalMs: 20_000,
  /** Hard cap so a rogue frame can't wedge the socket. */
  maxPayloadBytes: 6 * 1024 * 1024,
  jpegQuality: 0.72,
  maxFrameWidth: 1280,
  maxDomNodes: 220,
} as const;

const ACTION_KINDS: readonly ActionKind[] = [
  'click',
  'fill',
  'scroll',
  'navigate',
  'wait',
  'escalate',
  'done',
];

export function isActionKind(value: unknown): value is ActionKind {
  return typeof value === 'string' && (ACTION_KINDS as readonly string[]).includes(value);
}

export function isAgentAction(value: unknown): value is AgentAction {
  if (typeof value !== 'object' || value === null) return false;
  const a = value as Record<string, unknown>;
  if (!isActionKind(a.action)) return false;
  switch (a.action) {
    case 'click':
      return typeof a.selector === 'string' && a.selector.length > 0;
    case 'fill':
      return typeof a.selector === 'string' && typeof a.valueType === 'string';
    case 'navigate':
      return typeof a.url === 'string' && /^https?:\/\//i.test(a.url);
    case 'wait':
      return typeof a.ms === 'number' && Number.isFinite(a.ms);
    case 'scroll':
    case 'escalate':
    case 'done':
      return true;
    default:
      return false;
  }
}

export function isEnvelope(value: unknown): value is WsEnvelope<WsMessageType, unknown> {
  if (typeof value !== 'object' || value === null) return false;
  const e = value as Record<string, unknown>;
  return (
    e.v === PROTOCOL_VERSION &&
    typeof e.type === 'string' &&
    typeof e.id === 'string' &&
    typeof e.ts === 'number' &&
    'payload' in e
  );
}

export function envelope<T extends WsMessageType, P>(
  type: T,
  payload: P,
  id: string = newId(),
): WsEnvelope<T, P> {
  return { v: PROTOCOL_VERSION, type, id, ts: Date.now(), payload };
}

export function newId(): string {
  const g = globalThis as { crypto?: { randomUUID?: () => string } };
  if (g.crypto?.randomUUID) return g.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/* ------------------------------------------------------------------ *
 * Extension-internal messages (content <-> background <-> popup)
 * ------------------------------------------------------------------ */

export type RuntimeMessage =
  | { kind: 'AGENT_START'; goal: string; tabId?: number }
  | { kind: 'AGENT_STOP' }
  | { kind: 'AGENT_STATUS_REQUEST' }
  | { kind: 'AGENT_STATUS'; status: AgentStatus }
  | { kind: 'CAPTURE_FRAME_REQUEST' }
  | { kind: 'CAPTURE_FRAME_RESULT'; dataUrl: string }
  | { kind: 'ESCALATE'; request: InferenceRequestPayload }
  | { kind: 'ESCALATE_RESULT'; decision: AgentDecision | null; error?: string }
  | { kind: 'LOG'; entry: AgentLogEntry };

export interface AgentStatus {
  running: boolean;
  goal?: string;
  step: number;
  wsConnected: boolean;
  webgpuAvailable: boolean;
  localModelReady: boolean;
  localModelId?: string;
  lastDecision?: AgentDecision;
  escalations: number;
  localDecisions: number;
  redactions: number;
}

export interface AgentLogEntry {
  ts: number;
  level: 'debug' | 'info' | 'warn' | 'error';
  scope: string;
  message: string;
  data?: unknown;
}
