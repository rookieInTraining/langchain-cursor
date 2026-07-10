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

- Every generation spins up a **fresh throwaway local agent** (`Agent.create({ local: ... })`) with an empty scratch workspace, no user setting sources, an isolated run store, and — where the environment supports it — the SDK sandbox. The agent is closed after the call.
- The LangChain message history is serialized into a single role-tagged prompt (`<system>`, `<user>`, `<assistant>`, `<tool_result>`), and guardrails instruct the agent to behave as a plain chat model: no file access, no terminal, no MCP tools.
- **Tool calling is emulated through a prompt contract**: bound tool schemas are embedded in the prompt and the model must reply with a single `{"tool_calls": [...]}` JSON object, which is parsed back into LangChain `tool_calls`. A malformed reply triggers one corrective re-ask before failing.
- `withStructuredOutput` builds on the same mechanism and works with cached responses (plain `AIMessage`s, not just message chunks).
- Base64 `data:` images in `image_url` content blocks are forwarded to the agent as attachments (used for vision prompts).

Because a full agent is created per call, expect **higher latency** than direct API providers.

### Sandboxing

The SDK sandbox is requested by default. Some environments (e.g. WSL) do not support it — in that case `ChatCursor` logs a warning and permanently retries without the sandbox for that instance. The empty workspace, disabled setting sources, and prompt guardrails still apply.

### Timeouts, Aborts, and Retryable Errors

- Standard LangChain call options `signal` and `timeout` are honored; an aborted run is cancelled on the Cursor side.
- Errors keep (or are mapped to) names that common retry wrappers understand: the SDK's `RateLimitError` passes through, `AbortSignal.timeout` produces `TimeoutError`, and SDK errors flagged `isRetryable` (e.g. transient network failures) are wrapped as `TimeoutError`.

## Supported Models

Any model available to Cursor Agents. The default is `composer-2.5`.

## Limitations

- Requires a JavaScript runtime that resolves `node_modules` at runtime: `@cursor/sdk` ships a webpack-chunked dist that cannot be inlined, so single-file compiled binaries (e.g. `bun build --compile`) are not supported out of the box. Hosts that provision the SDK themselves can supply a custom `sdkLoader` (see API below).
- Streaming is not implemented; responses are returned when the agent run finishes.
- Node.js >= 20.

## API

### `new ChatCursor(fields?)`

Accepts all `BaseChatModel` params (e.g. `cache`, `callbacks`), plus:

- **`model`** — Cursor model id. Defaults to `composer-2.5`.
- **`apiKey`** — Cursor API key. Defaults to the `CURSOR_API_KEY` environment variable.
- **`sdkLoader`** — `() => Promise<CursorSdkModule>`. Overrides how `@cursor/sdk` is loaded. Useful when the runtime cannot resolve `node_modules` (e.g. single-file compiled binaries) and the host application provisions the SDK on its own — for example by downloading it from the npm registry at runtime and loading it with `createRequire`. Loader errors surface unwrapped.

## License

MIT
