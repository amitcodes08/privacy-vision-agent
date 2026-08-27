# Privacy Vision Agent

A client-dominant, privacy-preserving hybrid vision agent with hierarchical multi-step planning.

1. **Client-Side Query Planning**: Chrome Built-in **Gemini Nano (`ai.languageModel`)** decomposes complex, multi-clause user goals into an ordered sequence of atomic sub-objectives (with zero-overhead offline linguistic fallback).
2. **On-Device Vision Grounding**: A quantized **SmolVLM-256M-Instruct** model runs **inside the browser on WebGPU** to visually ground and execute each atomic sub-objective with zero network calls.
3. **Deterministic Ranker & Corroborator**: Grounds prompt element budgets and validates model intent against the active sub-goal.
4. **Termination Safeguards**: Validates progress against the entire sub-objective plan, preventing premature termination on intermediate search results.
5. **Redacted Cloud Escalation**: When the local model produces nothing usable, the extension escalates by sending a **redacted** screenshot (PII blacked out) plus a **scrubbed** DOM to a WebSocket server, which returns a structural command re-hydrated locally.

```
                      User Goal (e.g. "Search for shoes and add to cart")
                                              │
                                              ▼
                         [Gemini Nano / Query Decomposer]
                                              │
                                              ▼
                              Ordered Sub-Objectives Checklist
                                              │
content script ──scrubbed DOM + sensitive boxes──┤
                                              ▼
                           keyword ranker ──grounds active sub-goal──┐
                                     │                               ▼
captureVisibleTab ──raw frame──► offscreen ► WebGPU worker ► SmolVLM-256M plans action
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
                      ws://localhost:8080 ► cloud VLM (Gemini / Ollama)
                                     ▼
                  {action, selector, valueType} ► local hydration ► action        (3: last resort)
                                     │
                                     ▼
                      State Changed? Advance Sub-Objective ──► Repeat / Done
```

**The VLM is the planner for each atomic sub-goal.** The ranker never overrides a model that actually chose an element. Tier 3 (cloud escalation) acts strictly as a fallback when on-device models are unsure.

## Layout

| Path | Role |
| --- | --- |
| [shared/types.ts](shared/types.ts) | Wire contract: actions, envelopes, scrubbed DOM, `TaskMemory` with sub-objectives |
| [client/src/ai/nano-query-planner.ts](client/src/ai/nano-query-planner.ts) | Chrome Gemini Nano (`ai.languageModel`) query decomposer + rule-based fallback |
| [client/src/ai/termination-checker.ts](client/src/ai/termination-checker.ts) | Multi-step completion validator; guards against premature termination |
| [client/src/content/dom-scrubber.ts](client/src/content/dom-scrubber.ts) | Page → `ScrubbedDom` + boxes to black out |
| [client/src/privacy/pii-detector.ts](client/src/privacy/pii-detector.ts) | Luhn/Verhoeff-checked PII rules, deterministic |
| [client/src/privacy/canvas-redactor.ts](client/src/privacy/canvas-redactor.ts) | Destructive box painting → JPEG base64 |
| [client/src/ai/models.ts](client/src/ai/models.ts) | Model catalogue (`SmolVLM-256M` default) + **per-graph** quantization |
| [client/src/ai/model-loader.ts](client/src/ai/model-loader.ts) | Transformers.js v3 load + generate |
| [client/src/ai/ort-assets.ts](client/src/ai/ort-assets.ts) | Where onnxruntime may load WASM from, and in which shape |
| [client/src/ai/decision-parser.ts](client/src/ai/decision-parser.ts) | Prompt building, model text → action, selector repair + confidence |
| [client/src/ai/local-planner.ts](client/src/ai/local-planner.ts) | Keyword ranker: grounds active sub-goal in prompt and corroborates |
| [client/src/ai/vlm-worker.ts](client/src/ai/vlm-worker.ts) | Worker that owns the WebGPU context and weights |
| [client/scripts/copy-ort.mjs](client/scripts/copy-ort.mjs) | Vendors onnxruntime's WASM so MV3 never fetches code remotely |
| [client/src/offscreen/main.ts](client/src/offscreen/main.ts) | Offscreen host — the SW cannot own a GPU context |
| [client/src/background/index.ts](client/src/background/index.ts) | Orchestrator: query planning, sub-objective progression, escalation ladder |
| [client/src/network/ws-client.ts](client/src/network/ws-client.ts) | One socket, bounded reconnect ladder, request correlation |
| [client/src/content/value-hydrator.ts](client/src/content/value-hydrator.ts) | Resolves `USER_EMAIL` etc. from local storage |
| [server/src/websocket/handler.ts](server/src/websocket/handler.ts) | Envelope validation, rate limit, latency budget |
| [server/src/ai/cloud-vlm.ts](server/src/ai/cloud-vlm.ts) | Local Ollama / Gemini adapter + offline heuristic planner |

## Run it

```bash
npm install                      # workspace root
cp server/.env.example server/.env   # configured for Ollama by default
npm run dev:server               # ws://127.0.0.1:8080 (+ /health, /stats)
npm run dev:client               # runs copy:ort, then vite with HMR
```

`dev:client` and `build` both run [`scripts/copy-ort.mjs`](client/scripts/copy-ort.mjs)
first, which vendors onnxruntime's WASM into `client/public/ort/`. That is not
an optimisation — see [Local inference](#local-inference-and-what-it-took-to-make-it-actually-run)
below. If you invoke `vite` directly, run `npm run copy:ort --workspace client`
once yourself, or the local model will fail to load.

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

Six things had to be right before the on-device model produced a usable
action rather than silently deferring to the server:

- **Transformers.js fetches onnxruntime's WASM from a CDN, and MV3 forbids
  that.** `backends/onnx.js` points `wasm.wasmPaths` at
  `cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/` for anything
  that is not a `ServiceWorkerGlobalScope`. That is remotely hosted code:
  `script-src` in `content_security_policy.extension_pages` cannot whitelist a
  remote host, so `from_pretrained` threw before reading a single weight.
  `copy:ort` vendors the binary into the extension and `ort-assets.ts` repoints
  `wasmPaths` at the extension's own origin.
- **`wasmPaths` is a union, and the two arms take different code paths.** The
  obvious repair — `wasmPaths = 'chrome-extension://…/ort/'` — swapped the CDN
  failure for a stranger one. onnxruntime resolves for browsers to
  `ort.bundle.min.mjs`, the flavour that *embeds* the emscripten module factory
  precisely so nothing has to be fetched, and it uses that embedded copy only
  when neither a string prefix nor an `mjs` override is set:

  ```js
  if (!mjsUrl && !prefix && embedded && scriptUrl && sameOrigin(scriptUrl))
    return [undefined, embedded];
  // …otherwise: (await import(prefix + 'ort-wasm-simd-threaded.jsep.mjs')).default
  ```

  A string prefix therefore forces a dynamic import, and when that import
  resolves to something with no `default`, the factory is `undefined` and
  calling it throws `TypeError: f is not a function`. onnxruntime reports that as
  `no available backend found. ERR: [webgpu] …` — a WebGPU-shaped message for a
  failure that never reached WebGPU, and the extension's own error hint then
  agreed with it and blamed the shaders. The fix is the object form,
  `{ wasm: <url> }`, which is what onnxruntime itself passes to its proxy
  worker: the embedded factory is used, no `.mjs` is imported, and `locateFile`
  fetches the one vendored binary. `tests/ort-assets.test.ts` pins the shape,
  because the string form is the natural thing to write.
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
- **Show the model the element it needs.** `dom.nodes` holds up to 120
  entries, so listing the first N in DOM order could omit the one element the
  goal was about — the model then had no way to answer and looked "unsure" for
  a reason that was our fault. The ranker reserves slots for goal-relevant
  elements; the rest of the budget is page context. Emission stays in DOM order
  so the list lines up with the screenshot.

Two smaller things were making the same failure harder to read than it needed
to be. `chrome.offscreen.createDocument` resolves when the *document* exists,
which is not when its message listener is registered — the module script still
has to load, and in dev it loads over HTTP from the crxjs dev server. The first
`PROBE` lost that race, came back `Could not establish connection. Receiving end
does not exist.`, and warm-up reported it as a model failure; `askOffscreen` now
retries that specific error while the listener comes up. And a hard load failure
is remembered, so `runAgent` stops re-entering a download that already failed
structurally and re-printing the same multi-line error on every run. The popup's
explicit **Load model** still forces a fresh attempt.

That last point has a twin one layer down, and it is the one that made "click
on package.json" click something else. `buildScrubbedDom` also has a 120-node
budget, and it too spent it in document order — while its selector matches
`[role]`, which on a real application is nearly everything. The first 120
matches were the skip links, the logo, the global nav and the tab strip, so
`package.json` never entered `dom.nodes` at all and *neither* planner could
choose it. It now selects by priority — in-viewport dominates, because the
screenshot only shows the viewport, then operability, then having an accessible
name — and re-sorts into document order so ids stay monotonic. Operation verbs
(`click`, `press`, `type`) are also stopwords in the ranker now: they are
already read as intent, and as content keywords they scored every element on
the page that said "Click here".

The prompt asks the model for a numeric element **id** rather than a CSS
selector, and [decision-parser.ts](client/src/ai/decision-parser.ts) resolves
ids, exact selectors, `name` attributes, and labels back to a real element
before scoring. Recovering a near-miss is one escalation that does not happen.

## Rate limits, and keeping one socket

Two runtime budgets are easy to blow past, and both showed up as something
else entirely.

**`captureVisibleTab` allows about two calls per second.** The step loop took a
frame unconditionally, so a fast run hit
`MAX_CAPTURE_VISIBLE_TAB_CALLS_PER_SECOND` — and the rejection propagated out
of `step()` into `runAgent`'s catch, which killed the whole run rather than the
step. Three changes: capture is **lazy** (the ranker reads only the scrubbed
DOM, so a run with no model now takes zero screenshots), serialised behind a
600 ms gate, and retried once on the quota error specifically.

**One `WsClient`, one socket.** `connect()` used to return early only for
`open` and `connecting`, ignoring a *pending* reconnect — and the orchestrator
called it on every escalating step. So each call built a second socket while the
first was still live with its timer armed, and the abandoned socket kept its
listeners, so its close scheduled a reconnect too. That is the log showing
`reconnect #7` and `#8` inside the same second, with the backoff running away
to 36 s. Now: the guard covers a queued retry, `open`/`close`/`error` ignore any
socket that is not the current one, the ladder is capped at five attempts with
an 8 s ceiling and then goes **dormant** (a real `infer()` re-arms it), two
consecutive escalation failures trip a per-run circuit breaker, and the run's
`finally` closes the socket. `tests/ws-client.test.ts` drives this against a
fake socket rather than trusting the reading.

## Verification status

```
client: 129 passed, 2 skipped  (vitest, jsdom)
server: 8 passed               (vitest)
tsc --noEmit: clean in both workspaces
vite build:   extension bundles; copy:ort vendors 20.6 MB of ORT WASM into dist/ort/
smoke:        WS round trip 4-5 ms with VLM_PROVIDER=heuristic
```

With Ollama configured the smoke round trip is dominated by the model, not the
transport: a cold `qwen2.5vl:7b` (6 GB) does not finish loading inside
`OLLAMA_TIMEOUT_MS`, so the call aborts at 25 s and the heuristic planner
answers instead. That is the intended shape — the abort exists so the fallback
lands *inside* the client's 60 s escalation timeout rather than after it. A warm
model answers normally. Note the smoke fixture's 1×1 JPEG is rejected by
`qwen2.5vl` regardless of load state, so exercising the real Ollama vision path
needs a realistic frame.

The two skipped tests assert the <30 ms redaction budget; they need a real
`OffscreenCanvas`, so they only run in a browser-backed test environment.
The cloud path has **not** been exercised against a live Gemini key.

The local VLM load path *has* been run in a real Chrome profile with WebGPU —
that is where the `wasmPaths` union and the offscreen handshake race came from,
neither of which any amount of unit testing would have surfaced. The branch the
fix depends on was then verified in the shipped bundle rather than only in
`node_modules`: `dist/assets/vlm-worker-*.js` still contains the embedded
emscripten factory and `importWasmModule`'s `!mjsUrl && !prefix` fast path, and
the chunk carries `{wasmPaths:{wasm:…},numThreads:1}`. What remains unverified
is the step *after* loading — whether SmolVLM-256M, once resident, returns a
parseable action often enough to earn tier 1. That needs a manual pass.

## Privacy properties, and where they are enforced

- The unredacted frame reaches exactly one consumer: the offscreen WebGPU
  worker. `background/index.ts` hands `WsClient` only `redactFrame()` output.
  The downscale in `downscaleFrame()` also stays on the local path.
- The ranker is a pure function of the already-scrubbed DOM, so both the
  prompt grounding and the tier-2 fallback are the most private paths, not
  just the cheapest.
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
- The keyword ranker is synonym-driven. It grounds the prompt well for consent
  banners, logins, and search boxes, but it cannot rank an element whose
  wording shares nothing with the goal — such an element still reaches the
  prompt only as page-context filler, and the VLM has to spot it from the
  screenshot.
- `sharp` ships two high-severity libvips advisories as a transitive,
  Node-only dependency of `@huggingface/transformers`; it is not part of the
  extension bundle.
- The build emits ORT's WASM twice: once verbatim as `dist/ort/` (what
  `wasmPaths` points at, and the copy that is actually used) and once as a
  hashed `dist/assets/*.wasm` that Vite derives from the `new URL('…jsep.wasm',
  import.meta.url)` inside onnxruntime's *embedded* emscripten glue — the
  fallback `locateFile` would have used had we not overridden it. That is ~21 MB
  of dead weight in the unpacked extension. The clean removal is to point
  `wasmPaths.wasm` at that hashed asset instead, via a
  `?url` import, which would delete `copy:ort` and `public/ort/` outright. It is
  also a change to the exact resolution path that took two sessions to get
  right, so it waits until the load path has a browser-backed test.
- `numThreads` is pinned to 1. The embedded glue would spawn its pthread pool
  from its own `import.meta.url`, which after bundling is our worker chunk
  rather than a standalone emscripten module. WebGPU is unaffected; the wasm
  fallback gives up multi-core throughput for a load that completes.
