# Privacy Vision Agent

A client-dominant hybrid vision agent. A quantized VLM runs **inside the
browser on WebGPU** and acts on the page with zero network calls. Only when
the local model is not confident enough does the extension escalate — and
then it sends a **redacted** screenshot plus a **scrubbed** DOM to a
WebSocket server, which returns a *structural* command whose private values
are re-hydrated locally.

```
content script ──scrubbed DOM + sensitive boxes──┐
                                                 ▼
captureVisibleTab ──raw frame──► offscreen ► WebGPU worker ► action  (Path A: local, no network)
                                    │
                        confidence < threshold
                                    ▼
                      canvas redactor (black boxes)
                                    ▼
                     ws://localhost:8080 ► cloud VLM
                                    ▼
                 {action, selector, valueType} ► local hydration ► action  (Path B)
```

## Layout

| Path | Role |
| --- | --- |
| [shared/types.ts](shared/types.ts) | Wire contract: actions, envelopes, scrubbed DOM, guards |
| [client/src/content/dom-scrubber.ts](client/src/content/dom-scrubber.ts) | Page → `ScrubbedDom` + boxes to black out |
| [client/src/privacy/pii-detector.ts](client/src/privacy/pii-detector.ts) | Luhn/Verhoeff-checked PII rules, deterministic |
| [client/src/privacy/canvas-redactor.ts](client/src/privacy/canvas-redactor.ts) | Destructive box painting → JPEG base64 |
| [client/src/ai/vlm-worker.ts](client/src/ai/vlm-worker.ts) | Transformers.js v3 on WebGPU, in a worker |
| [client/src/offscreen/main.ts](client/src/offscreen/main.ts) | Offscreen host — the SW cannot own a GPU context |
| [client/src/background/index.ts](client/src/background/index.ts) | Orchestrator: the escalation gate lives here |
| [client/src/content/value-hydrator.ts](client/src/content/value-hydrator.ts) | Resolves `USER_EMAIL` etc. from local storage |
| [server/src/websocket/handler.ts](server/src/websocket/handler.ts) | Envelope validation, rate limit, latency budget |
| [server/src/ai/cloud-vlm.ts](server/src/ai/cloud-vlm.ts) | Local Ollama / Gemini adapter + offline heuristic planner |

## Run it

```bash
npm install                      # workspace root
cp server/.env.example server/.env   # configured for Ollama by default
npm run dev:server               # ws://127.0.0.1:8080 (+ /health, /stats)
npm run dev:client               # writes client/dist, HMR for the extension
```

### Running with Ollama (Local Server VLM):
Ensure Ollama is running on your machine with a vision model:
```bash
ollama run llama3.2-vision
# or: ollama run qwen2-vl
```

Then load `client/dist` via `chrome://extensions` → *Load unpacked*. Open the
popup, click **Load local model** (first run downloads ~230 MB of weights
from the Hub into the browser cache), type a goal, and hit **Run agent**.

If Ollama is offline or no model is configured, the server gracefully falls back to
its deterministic keyword planner so the entire loop is always demoable offline:

```bash
npm run smoke --workspace server   # CONNECT + INFERENCE_REQUEST round trip
```

## Verification status

```
client: 22 passed, 2 skipped   (vitest, jsdom)
server:  6 passed              (vitest)
tsc --noEmit: clean in both workspaces
vite build:   extension bundles (transformers/onnxruntime isolated to the worker chunk)
smoke:        WS round trip 4-5 ms against the heuristic planner
```

The two skipped tests assert the <30 ms redaction budget; they need a real
`OffscreenCanvas`, so they only run in a browser-backed test environment.
The cloud path has **not** been exercised against a live Gemini key, and the
local VLM path has not been run in a real Chrome profile with WebGPU — both
need a manual pass.

## Privacy properties, and where they are enforced

- The unredacted frame reaches exactly one consumer: the offscreen WebGPU
  worker. `background/index.ts` hands `WsClient` only `redactFrame()` output.
- Sensitive field values are replaced before serialization, not after:
  `dom-scrubber.ts` never copies a password, card, CVV, or OTP value into the
  payload; free text is run through `redactText()`.
- URLs are reduced to `origin + pathname`, dropping token-bearing queries.
- The server may name a value (`USER_EMAIL`) but never receives one.
  `value-hydrator.ts` **refuses** `USER_PASSWORD` and `OTP_CODE` outright, so
  a compromised server cannot make the extension type a credential.
- `redactionSummary` gives the popup a per-run privacy receipt.

## Known gaps

- **No auth on the WebSocket.** The server binds to `127.0.0.1` for that
  reason. Add a shared-secret handshake before exposing it off-host.
- Visual PII redaction is DOM-driven. Faces and PII baked into images are not
  detected yet; the `pii-detector` seam is where a local face detector goes.
- `parseAction` scores confidence structurally (does the selector exist?)
  rather than from token logprobs. Wiring `output_scores` in would make the
  gate sharper.
- `sharp` ships two high-severity libvips advisories as a transitive,
  Node-only dependency of `@huggingface/transformers`; it is not part of the
  extension bundle.
