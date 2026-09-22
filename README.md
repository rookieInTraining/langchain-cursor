# langchain-cursor

LangChain chat model that runs prompts through [Cursor Agents](https://cursor.com/docs/cloud-agent) (local runtime) using the official [`@cursor/sdk`](https://www.npmjs.com/package/@cursor/sdk), consuming your Cursor subscription instead of a direct model-provider API key.

## Installation

```bash
npm install @alumnium/langchain-cursor @langchain/core
```

Generate an API key in the [Cursor Dashboard](https://cursor.com/dashboard) and expose it as `CURSOR_API_KEY` (or pass it via the constructor).

## Usage

### Basic

```typescript
import { ChatCursor } from "@alumnium/langchain-cursor";

const llm = new ChatCursor(); // defaults to composer-2.5, reads CURSOR_API_KEY
const response = await llm.invoke("Hello!");
console.log(response.content);
```

### With a Specific Model or Explicit Key

```typescript
const llm = new ChatCursor({ model: "composer-2.5", apiKey: "..." });
```

### Tool Calling

```typescript
const llmWithTools = llm.bindTools([
  {
    type: "function",
    function: {
      name: "click",
      description: "Click an element",
      parameters: {
        type: "object",
        properties: { id: { type: "number" } },
        required: ["id"],
      },
    },
  },
]);

const message = await llmWithTools.invoke("Click the login button (id 7).");
console.log(message.tool_calls);
```

### Structured Output

```typescript
import { z } from "zod";

const structured = llm.withStructuredOutput(
  z.object({
    steps: z.array(z.string()),
    answer: z.string(),
  }),
);

const result = await structured.invoke("How to make coffee?");
console.log(result.steps);
```

## How It Works

Cursor exposes no chat-completions API — only the agent-shaped SDK. `ChatCursor` bridges the gap:

- Every generation spins up a **fresh throwaway local agent** with an empty scratch workspace, no user setting sources, an isolated run store, and — where the environment supports it — the SDK sandbox. The agent is closed after the call, so no conversation state leaks between invocations.
- The agent is created with **no built-in tools** (`tools: []`), so it cannot read files, run commands, search the web, or reach MCP servers. This is enforced by the Cursor backend rather than asked for in the prompt.
- LangChain `SystemMessage`s become the agent's own **`systemPrompt`**, replacing Cursor's built-in coding-assistant prompt. The remaining history is serialized into a single role-tagged prompt (`<user>`, `<assistant>`, `<tool_result>`).
- **Tool calling is emulated through a prompt contract**: bound tool schemas are embedded in the prompt and the model must reply with a single `{"tool_calls": [...]}` JSON object, which is parsed back into LangChain `tool_calls`. A malformed reply triggers one corrective re-ask before failing.
- `withStructuredOutput` builds on the same mechanism and works with cached responses (plain `AIMessage`s, not just message chunks).
- Base64 `data:` images in `image_url` content blocks are forwarded to the agent as attachments (used for vision prompts).

### Latency and the Warm Executor

The agent is cheap; the **local executor** behind it is not. Building it means authenticating, fetching feature gates, resolving sandbox policy, and scanning the workspace — and the SDK reference-counts it, tearing it down as soon as the last agent closes.

`ChatCursor` therefore takes a prewarm lease on the executor while each finished agent is still open, so the reference count never reaches zero between calls. An idle instance drops the lease after `keepAliveMs` (default 60s) and rebuilds on the next call.

Measured against `composer-2.5` with a trivial prompt, fresh process per row:

| | call 1 | later calls | steady-state avg |
|---|---|---|---|
| `keepAliveMs: 0` | 9943ms | 2581, 2685, 5025ms | 3430ms |
| `keepAliveMs: 60000` (default) | **7271ms** | 2041, 1945, 1990ms | **1992ms** |

The lease cuts steady-state latency by roughly 40% and, just as usefully, makes it predictable — a ~100ms spread instead of ~2400ms.

The first call is also cheaper, because the executor build is started *alongside* agent creation rather than after it. Those two phases do unrelated work — a workspace and auth bootstrap versus a model-catalog lookup — so overlapping them saves around 2.7s:

| phase | cost |
|---|---|
| `import("@cursor/sdk")` | ~180ms |
| executor build (auth, feature gates, sandbox policy, workspace scan) | ~4200ms |
| model catalog fetch | ~1300ms — overlapped, so mostly free |
| first agent turn | ~2600ms |

What remains is the ~4.2s executor build, which is network-bound inside the SDK. Nothing here removes it; it can only be moved off the critical path by constructing the model and issuing a throwaway call during your own startup rather than inside the first real request.

Two smaller levers, if you need them. Setting `CURSOR_SDK_LOCAL_MODEL_CATALOG_JSON` to a cached `Cursor.models.list()` result takes the catalog fetch from ~1300ms to ~30ms — useful across process restarts, though the overlap above already hides most of it. Node's `NODE_COMPILE_CACHE` is not worth it: ~36ms, since the cost is network and I/O rather than compilation.

The idle timer is `unref`ed, so a script that calls the model once and never disposes anything still exits normally. Long-lived hosts should call `await llm.dispose()` (or use `await using`) to release the executor as soon as the model is no longer needed.

Expect **higher latency than direct API providers** regardless: a warm executor removes per-call startup cost, not the agent round trip.

### Sandboxing

The SDK sandbox is requested by default. Some environments (e.g. WSL) do not support it — in that case `ChatCursor` logs a warning and permanently retries without the sandbox for that instance. The empty workspace, disabled setting sources, and the empty toolset still apply.

### Timeouts, Aborts, and Retryable Errors

- Standard LangChain call options `signal` and `timeout` are honored; an aborted run is cancelled on the Cursor side.
- Errors keep (or are mapped to) names that common retry wrappers understand: the SDK's `RateLimitError` passes through, `AbortSignal.timeout` produces `TimeoutError`, and SDK errors flagged `isRetryable` (e.g. transient network failures) are wrapped as `TimeoutError`.

## Supported Models

Any model available to Cursor Agents. The default is `composer-2.5`.

## Limitations

- Requires a JavaScript runtime that resolves `node_modules` at runtime: `@cursor/sdk` ships a webpack-chunked dist that cannot be inlined, so single-file compiled binaries (e.g. `bun build --compile`) are not supported out of the box. Hosts that provision the SDK themselves can supply a custom `sdkLoader` (see API below).
- Streaming is not implemented; responses are returned when the agent run finishes. (The SDK supports it — this package simply has not wired it up yet.)
- A custom system prompt is gated server-side. Accounts without access fall back automatically to inlining system messages in the prompt, with a warning logged once per instance.
- The run store grows for the life of an instance: one JSONL store serves every agent it creates.
- Node.js >= 20.

## API

### `new ChatCursor(fields?)`

Accepts all `BaseChatModel` params (e.g. `cache`, `callbacks`), plus:

- **`model`** — Cursor model id. Defaults to `composer-2.5`.
- **`apiKey`** — Cursor API key. Defaults to the `CURSOR_API_KEY` environment variable, then to any credentials stored by `Cursor.auth.login()`.
- **`sdkLoader`** — `() => Promise<CursorSdkModule>`. Overrides how `@cursor/sdk` is loaded. Useful when the runtime cannot resolve `node_modules` (e.g. single-file compiled binaries) and the host application provisions the SDK on its own — for example by downloading it from the npm registry at runtime and loading it with `createRequire`. Loader errors surface unwrapped. A module older than 1.0.32 still works: `ChatCursor` detects it and falls back to prompt-level guardrails and a per-call agent.
- **`keepAliveMs`** — how long an idle instance holds the warm Cursor executor, in milliseconds. Defaults to `60000`; `0` tears it down after every call.

### `chatCursor.dispose()`

Releases the warm executor immediately. Optional — an idle instance releases it on its own and never blocks process exit — but long-lived hosts should call it when the model is no longer needed. `ChatCursor` also implements `Symbol.asyncDispose`, so `await using llm = new ChatCursor()` works.

## License

MIT
