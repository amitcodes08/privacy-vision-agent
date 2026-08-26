/**
 * End-to-end smoke test for the escalation channel. Sends a CONNECT and a
 * realistic INFERENCE_REQUEST (already-redacted frame) and prints the
 * structural command that comes back.
 *
 *   node --experimental-strip-types scripts/smoke-client.ts
 */
import WebSocket from 'ws';
import { envelope, isEnvelope, type ServerMessage, type ScrubbedDom } from '../../shared/types.ts';

const URL = process.env.WS_URL ?? 'ws://127.0.0.1:8080';
const GOAL = process.argv[2] ?? 'accept the cookie banner';

const dom: ScrubbedDom = {
  url: 'https://shop.test/checkout',
  origin: 'https://shop.test',
  title: 'Checkout — shop.test',
  viewport: { width: 1280, height: 800, scrollX: 0, scrollY: 0 },
  nodes: [
    { id: 0, tag: 'button', selector: '#accept-cookies', text: 'Accept cookies', visible: true },
    { id: 1, tag: 'input', type: 'email', selector: '#email', label: 'Email address', visible: true },
    { id: 2, tag: 'input', type: 'password', selector: '#pw', label: 'Password', value: '[REDACTED]', visible: true, redacted: ['password'] },
  ],
  redactionSummary: { password: 1 },
};

// 1x1 black JPEG stands in for the redacted frame.
const FRAME =
  '/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNCwsNGRIT' +
  'DhQdGh8eHRocHCQuJyIkKikcHCctLTAwMzMzKz0/QD8/QD8/QD8/QD8/QD8/QD8/' +
  'QD8/QD8/QD8/QD8/QD8/QD8/QD8/QD//wAARCAABAAEDASIAAhEBAxEB/8QAFQAB' +
  'AQAAAAAAAAAAAAAAAAAAAAr/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAA' +
  'AAAAAAAAAAAAAAAAAAr/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEA' +
  'PwCdABmX/9k=';

const ws = new WebSocket(URL);
const started = Date.now();

ws.on('open', () => {
  console.log(`→ CONNECT ${URL}`);
  ws.send(
    JSON.stringify(
      envelope('CONNECT', {
        clientId: 'smoke-client',
        extensionVersion: '0.1.0',
        capabilities: { webgpu: false, localModelId: 'none' },
      }),
    ),
  );
  const req = envelope('INFERENCE_REQUEST', {
    goal: GOAL,
    imageBase64: FRAME,
    imageMime: 'image/jpeg' as const,
    dom,
    localConfidence: 0.21,
    localReason: 'below threshold',
  });
  console.log(`→ INFERENCE_REQUEST goal="${GOAL}"`);
  ws.send(JSON.stringify(req));
});

ws.on('message', (data) => {
  const parsed: unknown = JSON.parse(data.toString());
  if (!isEnvelope(parsed)) return console.error('← malformed frame');
  const msg = parsed as ServerMessage;
  console.log(`← ${msg.type}`, JSON.stringify(msg.payload).slice(0, 400));
  if (msg.type === 'INFERENCE_RESPONSE' || msg.type === 'ERROR') {
    console.log(`round trip: ${Date.now() - started}ms`);
    ws.close();
  }
});

ws.on('error', (err) => {
  console.error('socket error:', err.message);
  process.exit(1);
});
ws.on('close', () => process.exit(0));
