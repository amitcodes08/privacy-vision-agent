# Privacy Vision Agent

A client-dominant hybrid vision agent. A quantized VLM runs **inside the
browser on WebGPU** and decides what to do with zero network calls. A
deterministic keyword ranker keeps it honest: it picks which page elements the
model gets to look at, and afterwards corroborates whatever the model chose.
Only when the model produces nothing usable does the extension escalate — and
then it sends a **redacted** screenshot plus a **scrubbed** DOM to a WebSocket
server, which returns a *structural* command whose private values are
re-hydrated locally.

```
content script ──scrubbed DOM + sensitive boxes──┐
                                                 ▼
                          keyword ranker ──picks the elements──┐
                                    │                          ▼
captureVisibleTab ──raw frame──► offscreen ► WebGPU worker ► VLM plans the action
                                                               │
                                          ranker corroborates ─┤   (agreement raises confidence)
                                                               ▼
                                                             action              (1: local, no network)
                                    │
                       no usable model output
                                    ▼
                        ranker plans it instead ──► action                       (2: local, no network)
                                    │
                        VLM unsure about a real element
                                    ▼
                      canvas redactor (black boxes)
                                    ▼
                     ws://localhost:8080 ► cloud VLM
                                    ▼
                 {action, selector, valueType} ► local hydration ► action        (3: last resort)
```

**The VLM is the planner.** The ranker never overrides a model that actually
chose an element — a keyword match is not better evidence than a vision model
that read the page, and tier 3 exists for exactly that case. The ranker plans
alone only in tier 2, when there is no working model to plan with and the
alternative is nothing at all.

## Layout

| Path | Role |
| --- | --- |
| [shared/types.ts](shared/types.ts) | Wire contract: actions, envelopes, scrubbed DOM, guards |
| [client/src/content/dom-scrubber.ts](client/src/content/dom-scrubber.ts) | Page → `ScrubbedDom` + boxes to black out |
| [client/src/privacy/pii-detector.ts](client/src/privacy/pii-detector.ts) | Luhn/Verhoeff-checked PII rules, deterministic |
| [client/src/privacy/canvas-redactor.ts](client/src/privacy/canvas-redactor.ts) | Destructive box painting → JPEG base64 |
| [client/src/ai/models.ts](client/src/ai/models.ts) | Model catalogue + **per-graph** quantization |
| [client/src/ai/model-loader.ts](client/src/ai/model-loader.ts) | Transformers.js v3 load + generate |
| [client/src/ai/decision-parser.ts](client/src/ai/decision-parser.ts) | Prompt building, model text → action, selector repair + confidence |
| [client/src/ai/local-planner.ts](client/src/ai/local-planner.ts) | Keyword ranker: grounds the prompt, corroborates, and plans when there is no model |
| [client/src/ai/vlm-worker.ts](client/src/ai/vlm-worker.ts) | Worker that owns the WebGPU context and weights |
| [client/src/offscreen/main.ts](client/src/offscreen/main.ts) | Offscreen host — the SW cannot own a GPU context |
| [client/src/background/index.ts](client/src/background/index.ts) | Orchestrator: the escalation ladder lives here |
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
popup, type a goal, and hit **Run**. The local model starts downloading on
install (~230 MB of weights, cached by the browser afterwards); the popup's
status strip shows progress, and the on-device planner can already act while
it downloads. Auto-loading is a setting if you would rather click to load.

If Ollama is offline or no model is configured, the server gracefully falls back to
its deterministic keyword planner so the entire loop is always demoable offline:

```bash
npm run smoke --workspace server   # CONNECT + INFERENCE_REQUEST round trip
```

## Local inference, and what it took to make it actually run

Three things had to be right before the on-device model produced a usable
action rather than silently deferring to the server:

- **Quantization is per ONNX graph, not per model.** SmolVLM ships as
  `embed_tokens` + `vision_encoder` + `decoder_model_merged`. A flat
  `dtype: 'q4'` also 4-bit-quantizes the *embedding table*, and the decoder
  then emits fluent-looking garbage that never parses. `embed_tokens` stays
  fp16 (fp32 without `shader-f16`); the other two graphs take q4.
- **`AutoModelForImageTextToText`, not `AutoModelForVision2Seq`.** These are
  three-graph image-text-to-text models. Vision2Seq looks for an
  `encoder_model` the repos do not ship, and Qwen2-VL is not registered under
  it at all.
- **Decode only the generated tokens.** `generate()` returns prompt +
  completion. Cutting the prompt back off by string matching was unreliable,
  and on failure the JSON extractor latched onto the schema example *inside
  the prompt* — which parses, but is not a valid action. Slicing by
  `input_ids` length removes the failure mode.

The prompt asks the model for a numeric element **id** rather than a CSS
selector, and [decision-parser.ts](client/src/ai/decision-parser.ts) resolves
ids, exact selectors, `name` attributes, and labels back to a real element
before scoring. Recovering a near-miss is one escalation that does not happen.

## Verification status

```
client: 50 passed, 2 skipped   (vitest, jsdom)
server:  8 passed              (vitest)
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
  The downscale in `downscaleFrame()` also stays on the local path.
- Tier 2 is a pure function of the already-scrubbed DOM, so the cheapest
  fallback is also the most private one.
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
- Confidence is scored structurally (does the element resolve? is it enabled?)
  rather than from token logprobs. Wiring `output_scores` in would make the
  gate sharper.
- The tier-2 planner is keyword- and synonym-driven. It handles consent
  banners, logins, and search boxes well; it will not reason about multi-step
  intent, which is exactly what tiers 1 and 3 are for.
- `sharp` ships two high-severity libvips advisories as a transitive,
  Node-only dependency of `@huggingface/transformers`; it is not part of the
  extension bundle.
