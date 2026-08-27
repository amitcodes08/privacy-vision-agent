import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { envelope, type InferenceRequestPayload } from '@shared/types';
import { WsClient } from '~/network/ws-client';

/**
 * Minimal WebSocket stand-in. jsdom ships a real one, but these tests are about
 * *how many* sockets the client builds and when — so the socket has to be
 * inspectable and driven by hand.
 */
class FakeSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = FakeSocket.CONNECTING;
  readonly sent: string[] = [];
  closedWith: number | undefined;
  private readonly listeners = new Map<string, Set<(ev: unknown) => void>>();

  constructor(readonly url: string) {
    sockets.push(this);
  }

  addEventListener(type: string, fn: (ev: unknown) => void): void {
    let set = this.listeners.get(type);
    if (!set) this.listeners.set(type, (set = new Set()));
    set.add(fn);
  }
  removeEventListener(type: string, fn: (ev: unknown) => void): void {
    this.listeners.get(type)?.delete(fn);
  }
  send(data: string): void {
    this.sent.push(data);
  }
  close(code = 1000): void {
    this.readyState = FakeSocket.CLOSED;
    this.closedWith = code;
  }

  // ---- test drivers ----
  accept(): void {
    this.readyState = FakeSocket.OPEN;
    this.emit('open', {});
  }
  fail(code = 1006): void {
    this.emit('error', {});
    this.readyState = FakeSocket.CLOSED;
    this.emit('close', { code, reason: '' });
  }
  deliver(msg: unknown): void {
    this.emit('message', { data: JSON.stringify(msg) });
  }
  /** Envelope id of the nth frame this socket was asked to send. */
  frameId(n: number): string {
    return (JSON.parse(this.sent[n]!) as { id: string }).id;
  }

  private emit(type: string, ev: unknown): void {
    for (const fn of [...(this.listeners.get(type) ?? [])]) fn(ev);
  }
}

let sockets: FakeSocket[] = [];
const realWebSocket = globalThis.WebSocket;

/** Matches MAX_ATTEMPTS in ws-client.ts — the ladder is deliberately short. */
const LADDER = 5;

const REQUEST: InferenceRequestPayload = {
  goal: 'accept cookies',
  imageBase64: 'AAAA',
  imageMime: 'image/jpeg',
  dom: {
    url: 'https://shop.test/',
    origin: 'https://shop.test',
    title: 'Shop',
    viewport: { width: 1200, height: 800, scrollX: 0, scrollY: 0 },
    nodes: [],
    redactionSummary: {},
  },
};

const client = () => new WsClient({ url: 'ws://127.0.0.1:8080' });

/** Walk the whole reconnect ladder to exhaustion. */
async function exhaust(): Promise<void> {
  for (let i = 0; i < LADDER; i++) {
    sockets[i]!.fail();
    await vi.advanceTimersByTimeAsync(10_000);
  }
}

describe('WsClient reconnection', () => {
  beforeEach(() => {
    sockets = [];
    vi.useFakeTimers();
    globalThis.WebSocket = FakeSocket as unknown as typeof WebSocket;
  });

  afterEach(() => {
    vi.useRealTimers();
    globalThis.WebSocket = realWebSocket;
  });

  it('opens one socket no matter how often connect() is called', () => {
    const c = client();
    c.connect();
    c.connect();
    c.connect();
    expect(sockets).toHaveLength(1);
    c.close();
  });

  it('does not stack a second socket on top of a pending retry', async () => {
    const c = client();
    c.connect();
    sockets[0]!.fail();
    // Mid-backoff: this is the call `step()` used to make on every escalation.
    c.connect();
    c.connect();
    expect(sockets).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(sockets).toHaveLength(2);
    c.close();
  });

  it('counts one failed attempt per socket, not one per event', async () => {
    const states: string[] = [];
    const c = new WsClient({ url: 'ws://127.0.0.1:8080', onState: (s) => states.push(s) });
    c.connect();
    await exhaust();
    expect(sockets).toHaveLength(LADDER);
    expect(states.at(-1)).toBe('closed');
    c.close();
  });

  it('goes dormant instead of retrying forever', async () => {
    const c = client();
    c.connect();
    await exhaust();
    c.connect();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(sockets).toHaveLength(LADDER);
    c.close();
  });

  it('re-arms a dormant client for a real request, and completes it', async () => {
    const c = client();
    c.connect();
    await exhaust();

    const inflight = c.infer(REQUEST);
    expect(sockets).toHaveLength(LADDER + 1);

    const live = sockets[LADDER]!;
    live.accept();
    await vi.advanceTimersByTimeAsync(60);

    // sent[0] is CONNECT, sent[1] is the inference request.
    const decision = { action: { action: 'click' as const, selector: '#accept' }, confidence: 0.9, source: 'cloud' as const };
    live.deliver(envelope('INFERENCE_RESPONSE', { decision }, live.frameId(1)));

    await expect(inflight).resolves.toMatchObject({ decision });
    expect(c.connected).toBe(true);
    c.close();
  });

  it('rejects in-flight requests when the socket drops', async () => {
    const c = client();
    c.connect();
    sockets[0]!.accept();
    const inflight = c.infer(REQUEST);
    sockets[0]!.fail();
    await expect(inflight).rejects.toThrow(/socket closed/);
    c.close();
  });

  it('close() stops the ladder and resets the budget', async () => {
    const c = client();
    c.connect();
    sockets[0]!.fail();
    c.close();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(sockets).toHaveLength(1);
    // A fresh connect() after close() starts from attempt 0, not from dormant.
    c.connect();
    expect(sockets).toHaveLength(2);
    c.close();
  });
});
