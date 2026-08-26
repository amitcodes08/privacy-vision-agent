import 'dotenv/config';
import express from 'express';
import { createServer } from 'node:http';
import { attachWebSocket, getStats } from './websocket/handler.ts';
import { activeModelId } from './ai/cloud-vlm.ts';

const PORT = Number(process.env.PORT ?? 8080);
// Bound to loopback on purpose: the socket has NO authentication, so it must
// not be reachable off-host. Add a token check before changing this.
const HOST = process.env.HOST ?? '127.0.0.1';

const app = express();
app.disable('x-powered-by');
app.get('/health', (_req, res) => {
  res.json({ ok: true, model: activeModelId, uptimeSec: Math.round(process.uptime()) });
});
app.get('/stats', (_req, res) => res.json(getStats()));

const server = createServer(app);
attachWebSocket(server);

server.listen(PORT, HOST, () => {
  console.log(`[pva] escalation server on ws://${HOST}:${PORT} (model: ${activeModelId})`);
  if (activeModelId === 'heuristic-fallback') {
    console.warn('[pva] No VLM active — using the offline heuristic planner');
  }
});

for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, () => {
    console.log(`[pva] ${sig} — shutting down`);
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 2_000).unref();
  });
}
