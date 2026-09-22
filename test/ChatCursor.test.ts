import type { ToolDefinition } from "@langchain/core/language_models/base";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  type MockInstance,
  vi,
} from "vitest";
import z from "zod";
import { ChatCursor } from "../src/ChatCursor.js";

const createMock = vi.fn();
const sendMock = vi.fn();
const closeMock = vi.fn();
const cancelMock = vi.fn();
const platformMock = vi.fn();
const prewarmMock = vi.fn();
const releaseMock = vi.fn();
const storeMock = vi.fn();

// `createAgent` routes to the same spy as `Agent.create` so option
// assertions read the same regardless of which path built the agent.
vi.mock("@cursor/sdk", () => ({
  Agent: {
    create: (...args: unknown[]) => createMock(...args),
  },
  createAgentPlatform: (...args: unknown[]) => platformMock(...args),
  JsonlLocalAgentStore: class {
    rootDir: string;
    constructor(rootDir: string) {
      this.rootDir = rootDir;
      storeMock(rootDir);
    }
  },
}));

interface RunResultStub {
  id?: string;
  status?: string;
  result?: string;
  error?: { message: string; code?: string };
  usage?: Record<string, number>;
}

function stubRun(result: RunResultStub, waitOverride?: () => Promise<never>) {
  const runResult = { id: "run-1", status: "finished", ...result };
  return {
    id: runResult.id,
    agentId: "agent-1",
    wait: waitOverride
      ? vi.fn(waitOverride)
      : vi.fn().mockResolvedValue(runResult),
    cancel: cancelMock.mockResolvedValue(undefined),
  };
}

function stubAgent(...runs: ReturnType<typeof stubRun>[]) {
  for (const run of runs) sendMock.mockResolvedValueOnce(run);
  createMock.mockResolvedValue({
    agentId: "agent-1",
    send: sendMock,
    close: closeMock,
  });
}

const CLICK_TOOL: ToolDefinition = {
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
};

const MESSAGES = [
  new SystemMessage("You are a test agent."),
  new HumanMessage("Do the thing."),
];

let warnSpy: MockInstance;

beforeEach(() => {
  createMock.mockReset();
  sendMock.mockReset();
  closeMock.mockReset();
  cancelMock.mockReset();
  storeMock.mockReset();

  releaseMock.mockReset().mockResolvedValue(undefined);
  prewarmMock.mockReset().mockResolvedValue(releaseMock);
  platformMock.mockReset().mockImplementation(() =>
    Promise.resolve({
      createAgent: (...args: unknown[]) => createMock(...args),
      prewarmLocalWorkspace: (...args: unknown[]) => prewarmMock(...args),
    }),
  );

  warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  warnSpy.mockRestore();
  vi.unstubAllEnvs();
});

describe("ChatCursor._generate", () => {
  it("returns the run result text with usage and id", async () => {
    stubAgent(
      stubRun({
        id: "run-42",
        result: "The answer",
        usage: {
          inputTokens: 10,
          outputTokens: 5,
          cacheReadTokens: 2,
          cacheWriteTokens: 1,
          totalTokens: 15,
        },
      }),
    );

    const llm = new ChatCursor({ model: "composer-2.5", apiKey: "key" });
    const message = await llm.invoke(MESSAGES);

    expect(message.content).toBe("The answer");
    expect(message.id).toBe("run-42");
    expect(message.usage_metadata).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      input_token_details: { cache_read: 2, cache_creation: 1 },
    });
    expect(closeMock).toHaveBeenCalled();
  });

  it("creates a sandboxed local agent without user settings", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5", apiKey: "key" });
    await llm.invoke(MESSAGES);

    expect(createMock).toHaveBeenCalledWith(
      expect.objectContaining({
        apiKey: "key",
        model: { id: "composer-2.5" },
        local: expect.objectContaining({
          settingSources: [],
          sandboxOptions: { enabled: true },
        }),
      }),
    );
  });

  it("falls back to the CURSOR_API_KEY environment variable", async () => {
    vi.stubEnv("CURSOR_API_KEY", "env-key");
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);

    expect(createMock).toHaveBeenCalledWith(
      expect.objectContaining({ apiKey: "env-key" }),
    );
  });

  it("sends role-tagged prompt text", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke([
      new HumanMessage("Do the thing."),
      new AIMessage("Done."),
    ]);

    const payload = sendMock.mock.calls[0]?.[0];
    expect(payload.text).toContain("<user>\nDo the thing.\n</user>");
    expect(payload.text).toContain("<assistant>\nDone.\n</assistant>");
  });

  it("restricts the agent to a text-only toolset instead of prompt guardrails", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);

    expect(createMock).toHaveBeenCalledWith(
      expect.objectContaining({ tools: [] }),
    );
    const payload = sendMock.mock.calls[0]?.[0];
    expect(payload.text).not.toContain("# Execution context");
  });

  it("lifts system messages into the agent's system prompt", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);

    expect(createMock).toHaveBeenCalledWith(
      expect.objectContaining({ systemPrompt: "You are a test agent." }),
    );
    const payload = sendMock.mock.calls[0]?.[0];
    expect(payload.text).not.toContain("<system>");
    expect(payload.text).toContain("<user>\nDo the thing.\n</user>");
  });

  it("leaves systemPrompt unset when there are no system messages", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke([new HumanMessage("Do the thing.")]);

    expect(createMock.mock.calls[0]?.[0]).not.toHaveProperty("systemPrompt");
  });

  it("forwards data-URL images to the agent", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: "Describe" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,aGVsbG8=" },
          },
        ],
      }),
    ]);

    const payload = sendMock.mock.calls[0]?.[0];
    expect(payload.images).toEqual([
      { data: "aGVsbG8=", mimeType: "image/png" },
    ]);
  });

  it("closes the agent when the run fails", async () => {
    stubAgent(stubRun({ status: "error", error: { message: "boom" } }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await expect(llm.invoke(MESSAGES)).rejects.toThrow(/boom/);
    expect(closeMock).toHaveBeenCalled();
  });

  it("produces messages with the fields caching layers rely on", async () => {
    stubAgent(
      stubRun({
        result: "cached text",
        usage: {
          inputTokens: 1,
          outputTokens: 2,
          cacheReadTokens: 0,
          cacheWriteTokens: 0,
          totalTokens: 3,
        },
      }),
    );

    const llm = new ChatCursor({ model: "composer-2.5" });
    const message = (await llm.invoke(MESSAGES)) as AIMessage;

    // Cache schemas (e.g. Alumnium's Lchain) require these fields to be
    // present to serialize and deserialize generations losslessly.
    expect(message.id).toBe("run-1");
    expect(message.tool_calls).toEqual([]);
    expect(message.invalid_tool_calls ?? []).toEqual([]);
    expect(message.response_metadata).toMatchObject({
      model_name: "composer-2.5",
    });
    expect(message.usage_metadata).toBeDefined();
  });
});

describe("ChatCursor tool calling", () => {
  it("parses tool calls from the JSON reply", async () => {
    stubAgent(
      stubRun({
        result: '{"tool_calls": [{"name": "click", "arguments": {"id": 7}}]}',
      }),
    );

    const llm = new ChatCursor({ model: "composer-2.5" });
    const message = (await llm
      .bindTools([CLICK_TOOL])
      .invoke(MESSAGES)) as AIMessage;

    expect(message.tool_calls).toHaveLength(1);
    expect(message.tool_calls?.[0]).toMatchObject({
      name: "click",
      args: { id: 7 },
      type: "tool_call",
    });
    expect(message.tool_calls?.[0]?.id).toMatch(/^call_/);
    // The raw reply stays in content: callers may rely on generations having
    // non-empty content.
    expect(message.content).toBe(
      '{"tool_calls": [{"name": "click", "arguments": {"id": 7}}]}',
    );
  });

  it("embeds the tool contract in the prompt", async () => {
    stubAgent(stubRun({ result: '{"tool_calls": []}' }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.bindTools([CLICK_TOOL]).invoke(MESSAGES);

    const payload = sendMock.mock.calls[0]?.[0];
    expect(payload.text).toContain("# Tool calling");
    expect(payload.text).toContain('"click"');
  });

  it("re-asks once when the reply is not valid tool call JSON", async () => {
    stubAgent(
      stubRun({ result: "I clicked it for you!" }),
      stubRun({
        result: '{"tool_calls": [{"name": "click", "arguments": {"id": 1}}]}',
      }),
    );

    const llm = new ChatCursor({ model: "composer-2.5" });
    const message = (await llm
      .bindTools([CLICK_TOOL])
      .invoke(MESSAGES)) as AIMessage;

    expect(sendMock).toHaveBeenCalledTimes(2);
    expect(sendMock.mock.calls[1]?.[0].text).toMatch(/only.*json/i);
    expect(message.tool_calls?.[0]).toMatchObject({
      name: "click",
      args: { id: 1 },
    });
    expect(message.content).toBe(
      '{"tool_calls": [{"name": "click", "arguments": {"id": 1}}]}',
    );
  });

  it("throws when the corrective re-ask also fails", async () => {
    stubAgent(stubRun({ result: "prose" }), stubRun({ result: "more prose" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await expect(llm.bindTools([CLICK_TOOL]).invoke(MESSAGES)).rejects.toThrow(
      /JSON/,
    );
    expect(closeMock).toHaveBeenCalled();
  });
});

describe("ChatCursor sandbox fallback", () => {
  function sandboxUnsupportedError() {
    const error = new Error(
      "Local SDK sandboxing was requested, but sandboxing is not supported " +
        "in this environment. Disable local.sandboxOptions.enabled or remove " +
        "~/.cursor/sandbox.json to run without sandboxing.",
    );
    error.name = "ConfigurationError";
    return error;
  }

  it("retries without sandboxing when the environment does not support it", async () => {
    sendMock.mockRejectedValueOnce(sandboxUnsupportedError());
    sendMock.mockResolvedValueOnce(stubRun({ result: "recovered" }));
    createMock.mockResolvedValue({
      agentId: "agent-1",
      send: sendMock,
      close: closeMock,
    });

    const llm = new ChatCursor({ model: "composer-2.5" });
    const message = await llm.invoke(MESSAGES);

    expect(message.content).toBe("recovered");
    expect(createMock).toHaveBeenCalledTimes(2);
    expect(createMock.mock.calls[0]?.[0].local.sandboxOptions).toEqual({
      enabled: true,
    });
    expect(createMock.mock.calls[1]?.[0].local.sandboxOptions).toEqual({
      enabled: false,
    });
    expect(warnSpy).toHaveBeenCalledOnce();
    expect(warnSpy.mock.calls[0]?.[0]).toMatch(/sandbox/i);
  });

  it("skips sandboxing on subsequent calls once unsupported", async () => {
    sendMock.mockRejectedValueOnce(sandboxUnsupportedError());
    sendMock.mockResolvedValueOnce(stubRun({ result: "first" }));
    sendMock.mockResolvedValueOnce(stubRun({ result: "second" }));
    createMock.mockResolvedValue({
      agentId: "agent-1",
      send: sendMock,
      close: closeMock,
    });

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);
    await llm.invoke(MESSAGES);

    expect(createMock).toHaveBeenCalledTimes(3);
    expect(createMock.mock.calls[2]?.[0].local.sandboxOptions).toEqual({
      enabled: false,
    });
    expect(warnSpy).toHaveBeenCalledOnce();
  });

  it("propagates unrelated configuration errors", async () => {
    const error = new Error("Invalid model name");
    error.name = "ConfigurationError";
    sendMock.mockRejectedValueOnce(error);
    createMock.mockResolvedValue({
      agentId: "agent-1",
      send: sendMock,
      close: closeMock,
    });

    const llm = new ChatCursor({ model: "composer-2.5" });
    await expect(llm.invoke(MESSAGES)).rejects.toThrow(/Invalid model name/);
    expect(createMock).toHaveBeenCalledTimes(1);
  });
});

describe("ChatCursor error mapping", () => {
  it("preserves SDK rate limit errors for retry wrappers", async () => {
    const rateLimitError = new Error("slow down");
    rateLimitError.name = "RateLimitError";
    sendMock.mockRejectedValueOnce(rateLimitError);
    createMock.mockResolvedValue({
      agentId: "agent-1",
      send: sendMock,
      close: closeMock,
    });

    const llm = new ChatCursor({ model: "composer-2.5" });
    await expect(llm.invoke(MESSAGES)).rejects.toMatchObject({
      name: "RateLimitError",
    });
  });

  it("maps retryable SDK errors to TimeoutError", async () => {
    const networkError = Object.assign(new Error("service unavailable"), {
      name: "NetworkError",
      isRetryable: true,
    });
    sendMock.mockRejectedValueOnce(networkError);
    createMock.mockResolvedValue({
      agentId: "agent-1",
      send: sendMock,
      close: closeMock,
    });

    const llm = new ChatCursor({ model: "composer-2.5" });
    await expect(llm.invoke(MESSAGES)).rejects.toMatchObject({
      name: "TimeoutError",
    });
  });

  it("aborts a hung run with a TimeoutError and cancels it", async () => {
    stubAgent(stubRun({}, () => new Promise<never>(() => {})));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await expect(llm.invoke(MESSAGES, { timeout: 25 })).rejects.toMatchObject({
      name: "TimeoutError",
    });
    expect(cancelMock).toHaveBeenCalled();
    expect(closeMock).toHaveBeenCalled();
  });
});

describe("ChatCursor.withStructuredOutput", () => {
  const Plan = z.object({
    explanation: z.string(),
    actions: z.array(z.string()),
  });

  it("returns raw and parsed output from a live call", async () => {
    stubAgent(
      stubRun({
        result:
          '{"tool_calls": [{"name": "extract", "arguments": {"explanation": "why", "actions": ["a", "b"]}}]}',
      }),
    );

    const llm = new ChatCursor({ model: "composer-2.5" });
    const result = (await llm
      .withStructuredOutput(Plan, { includeRaw: true })
      .invoke(MESSAGES)) as { raw: AIMessage; parsed: z.infer<typeof Plan> };

    expect(result.parsed).toEqual({ explanation: "why", actions: ["a", "b"] });
    expect(result.raw.tool_calls?.[0]?.name).toBe("extract");
  });

  it("parses plain AIMessages from cache hits", async () => {
    const cachedMessage = new AIMessage({
      id: "cached-1",
      content: "",
      tool_calls: [
        {
          type: "tool_call",
          id: "call_cached",
          name: "extract",
          args: { explanation: "cached", actions: ["x"] },
        },
      ],
    });
    const cache = {
      lookup: vi.fn().mockResolvedValue([{ text: "", message: cachedMessage }]),
      update: vi.fn(),
    };

    const llm = new ChatCursor({
      model: "composer-2.5",
      cache: cache as never,
    });
    const result = (await llm
      .withStructuredOutput(Plan, { includeRaw: true })
      .invoke(MESSAGES)) as { raw: AIMessage; parsed: z.infer<typeof Plan> };

    expect(createMock).not.toHaveBeenCalled();
    expect(result.parsed).toEqual({ explanation: "cached", actions: ["x"] });
  });
});

describe("ChatCursor custom sdkLoader", () => {
  it("loads the SDK through the provided loader instead of import", async () => {
    const loaderSendMock = vi.fn().mockResolvedValue(
      stubRun({ id: "run-loader", result: "from loader" }),
    );
    const loaderCloseMock = vi.fn();
    const loaderCreateMock = vi.fn().mockResolvedValue({
      agentId: "agent-loader",
      send: loaderSendMock,
      close: loaderCloseMock,
    });
    const sdkLoader = vi.fn().mockResolvedValue({
      Agent: { create: loaderCreateMock },
      JsonlLocalAgentStore: class {
        rootDir: string;
        constructor(rootDir: string) {
          this.rootDir = rootDir;
        }
      },
    });

    const llm = new ChatCursor({
      model: "composer-2.5",
      apiKey: "key",
      sdkLoader: sdkLoader as never,
    });
    const message = await llm.invoke(MESSAGES);

    expect(sdkLoader).toHaveBeenCalled();
    expect(loaderCreateMock).toHaveBeenCalled();
    expect(message.content).toBe("from loader");
    expect(loaderCloseMock).toHaveBeenCalled();
    expect(createMock).not.toHaveBeenCalled();
  });

  it("surfaces loader errors unwrapped", async () => {
    const failure = new Error("vendor install failed: registry unreachable");
    const llm = new ChatCursor({
      model: "composer-2.5",
      apiKey: "key",
      sdkLoader: () => Promise.reject(failure),
    });

    await expect(llm.invoke(MESSAGES)).rejects.toBe(failure);
    expect(createMock).not.toHaveBeenCalled();
  });
});

describe("ChatCursor executor warmth", () => {
  it("takes the prewarm lease before closing the agent", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const order: string[] = [];
    prewarmMock.mockImplementation(() => {
      order.push("prewarm");
      return Promise.resolve(releaseMock);
    });
    closeMock.mockImplementation(() => order.push("close"));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);

    // Reversing these would drop the executor's reference count to zero and
    // let the SDK tear down the very thing being kept warm.
    expect(order).toEqual(["prewarm", "close"]);
  });

  it("holds one lease across calls and reuses the platform and store", async () => {
    stubAgent(stubRun({ result: "one" }), stubRun({ result: "two" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);
    await llm.invoke(MESSAGES);

    expect(createMock).toHaveBeenCalledTimes(2);
    expect(prewarmMock).toHaveBeenCalledTimes(1);
    expect(releaseMock).not.toHaveBeenCalled();
    expect(platformMock).toHaveBeenCalledTimes(1);
    expect(storeMock).toHaveBeenCalledTimes(1);
  });

  it("does not pass a per-agent store once the platform owns it", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);

    // The platform is constructed with the scratch store, which is what keeps
    // run records out of the user's Cursor state root.
    expect(platformMock).toHaveBeenCalledWith(
      expect.objectContaining({
        localStore: expect.objectContaining({ rootDir: expect.any(String) }),
      }),
    );
    expect(createMock.mock.calls[0]?.[0].local).not.toHaveProperty("store");
  });

  it("releases the lease when the instance is disposed", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);
    expect(releaseMock).not.toHaveBeenCalled();

    await llm.dispose();
    expect(releaseMock).toHaveBeenCalledTimes(1);
  });

  it("releases the lease once the keep-alive window lapses", async () => {
    vi.useFakeTimers();
    try {
      stubAgent(stubRun({ result: "ok" }));

      const llm = new ChatCursor({ model: "composer-2.5", keepAliveMs: 1000 });
      await llm.invoke(MESSAGES);
      expect(releaseMock).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(1001);
      expect(releaseMock).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("never warms the executor when keep-alive is disabled", async () => {
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5", keepAliveMs: 0 });
    await llm.invoke(MESSAGES);

    expect(prewarmMock).not.toHaveBeenCalled();
    await expect(llm.dispose()).resolves.toBeUndefined();
  });

  it("keeps generating when prewarming fails", async () => {
    stubAgent(stubRun({ result: "ok" }));
    prewarmMock.mockRejectedValue(new Error("prewarm exploded"));

    const llm = new ChatCursor({ model: "composer-2.5" });

    // Prewarming is a pure optimization; a failure must not surface.
    await expect(llm.invoke(MESSAGES)).resolves.toMatchObject({
      content: "ok",
    });
    expect(closeMock).toHaveBeenCalled();
  });

  it("re-warms under the new options after the sandbox falls back", async () => {
    const unsupported = new Error("sandboxOptions: sandboxing is unavailable");
    unsupported.name = "ConfigurationError";
    createMock.mockRejectedValueOnce(unsupported);
    sendMock.mockResolvedValueOnce(stubRun({ result: "recovered" }));
    createMock.mockResolvedValue({
      agentId: "agent-1",
      send: sendMock,
      close: closeMock,
    });

    const staleRelease = vi.fn().mockResolvedValue(undefined);
    prewarmMock.mockResolvedValueOnce(staleRelease);

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);

    // Warming starts alongside the agent, before the sandbox verdict is in,
    // so the first attempt warms the wrong key. That is the accepted cost of
    // overlapping the two: the retry warms the options that actually worked
    // and the stale lease is released rather than leaked.
    expect(prewarmMock).toHaveBeenCalledTimes(2);
    expect(prewarmMock.mock.calls[0]?.[0].local.sandboxOptions).toEqual({
      enabled: true,
    });
    expect(prewarmMock.mock.calls[1]?.[0].local.sandboxOptions).toEqual({
      enabled: false,
    });
    expect(staleRelease).toHaveBeenCalledTimes(1);
  });

  it("overlaps the executor build with agent creation", async () => {
    stubAgent(stubRun({ result: "ok" }));

    // Each side blocks until the other has started, so this can only finish
    // if both are genuinely in flight at once. Were they sequenced either
    // way, one would wait forever and the test would time out.
    let prewarmStarted = () => {};
    let agentStarted = () => {};
    const prewarmRunning = new Promise<void>((r) => (prewarmStarted = r));
    const agentRunning = new Promise<void>((r) => (agentStarted = r));

    prewarmMock.mockImplementation(async () => {
      prewarmStarted();
      await agentRunning;
      return releaseMock;
    });
    platformMock.mockImplementation(() =>
      Promise.resolve({
        createAgent: async (...args: unknown[]) => {
          agentStarted();
          await prewarmRunning;
          return createMock(...args);
        },
        prewarmLocalWorkspace: (...args: unknown[]) => prewarmMock(...args),
      }),
    );

    const llm = new ChatCursor({ model: "composer-2.5" });
    await expect(llm.invoke(MESSAGES)).resolves.toMatchObject({
      content: "ok",
    });
    expect(prewarmMock).toHaveBeenCalledTimes(1);
  });
});

describe("ChatCursor system prompt fallback", () => {
  function systemPromptGatedError() {
    const error = new Error(
      "[invalid_argument] --system-prompt is not enabled for this account",
    );
    error.name = "CursorAgentError";
    return error;
  }

  it("falls back to inline system messages and sticks to it", async () => {
    sendMock.mockRejectedValueOnce(systemPromptGatedError());
    sendMock.mockResolvedValueOnce(stubRun({ result: "inlined" }));
    sendMock.mockResolvedValueOnce(stubRun({ result: "second" }));
    createMock.mockResolvedValue({
      agentId: "agent-1",
      send: sendMock,
      close: closeMock,
    });

    const llm = new ChatCursor({ model: "composer-2.5" });
    const message = await llm.invoke(MESSAGES);

    expect(message.content).toBe("inlined");
    expect(sendMock.mock.calls[1]?.[0].text).toContain(
      "<system>\nYou are a test agent.\n</system>",
    );
    expect(createMock.mock.calls[1]?.[0]).not.toHaveProperty("systemPrompt");
    expect(warnSpy).toHaveBeenCalledTimes(1);

    await llm.invoke(MESSAGES);
    expect(createMock.mock.calls[2]?.[0]).not.toHaveProperty("systemPrompt");
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it("reports the original rejection when the fallback also fails", async () => {
    const gated = systemPromptGatedError();
    sendMock.mockRejectedValueOnce(gated);
    sendMock.mockRejectedValueOnce(new Error("something else entirely"));
    createMock.mockResolvedValue({
      agentId: "agent-1",
      send: sendMock,
      close: closeMock,
    });

    const llm = new ChatCursor({ model: "composer-2.5" });

    // The gate is only recognizable by its message text, so a match may be a
    // false positive — the first error is the trustworthy one.
    await expect(llm.invoke(MESSAGES)).rejects.toBe(gated);
  });

  it("leaves unrelated invalid-argument errors alone", async () => {
    sendMock.mockRejectedValue(new Error("[invalid_argument] bad model"));
    createMock.mockResolvedValue({
      agentId: "agent-1",
      send: sendMock,
      close: closeMock,
    });

    const llm = new ChatCursor({ model: "composer-2.5" });

    await expect(llm.invoke(MESSAGES)).rejects.toThrow(/bad model/);
    expect(createMock).toHaveBeenCalledTimes(1);
  });
});

describe("ChatCursor legacy SDK", () => {
  it("degrades to Agent.create with prompt guardrails", async () => {
    const legacySend = vi.fn().mockResolvedValue(stubRun({ result: "legacy" }));
    const legacyClose = vi.fn();
    const legacyCreate = vi.fn().mockResolvedValue({
      agentId: "agent-1",
      send: legacySend,
      close: legacyClose,
    });

    const llm = new ChatCursor({
      model: "composer-2.5",
      apiKey: "key",
      // A module predating `createAgentPlatform` also ignores `tools` and
      // `systemPrompt`, so the prompt has to carry the guardrails again.
      sdkLoader: (() =>
        Promise.resolve({
          Agent: { create: legacyCreate },
          JsonlLocalAgentStore: class {
            rootDir: string;
            constructor(rootDir: string) {
              this.rootDir = rootDir;
            }
          },
        })) as never,
    });
    const message = await llm.invoke(MESSAGES);

    expect(message.content).toBe("legacy");
    const options = legacyCreate.mock.calls[0]?.[0];
    expect(options).not.toHaveProperty("tools");
    expect(options).not.toHaveProperty("systemPrompt");
    expect(options.local).toHaveProperty("store");

    const payload = legacySend.mock.calls[0]?.[0];
    expect(payload.text).toContain("# Execution context");
    expect(payload.text).toContain("<system>\nYou are a test agent.\n</system>");
  });
});

describe("ChatCursor usage and cancellation", () => {
  it("sums usage across a corrective re-ask", async () => {
    stubAgent(
      stubRun({
        result: "not json at all",
        usage: {
          inputTokens: 10,
          outputTokens: 4,
          totalTokens: 14,
          cacheReadTokens: 1,
          cacheWriteTokens: 2,
        },
      }),
      stubRun({
        result: '{"tool_calls": [{"name": "click", "arguments": {"id": 1}}]}',
        usage: {
          inputTokens: 5,
          outputTokens: 3,
          totalTokens: 8,
          cacheReadTokens: 4,
          cacheWriteTokens: 0,
        },
      }),
    );

    const llm = new ChatCursor({ model: "composer-2.5" });
    const message = await llm.bindTools([CLICK_TOOL]).invoke(MESSAGES);

    // Both runs are billed, so the caller has to see both.
    expect(message.usage_metadata).toEqual({
      input_tokens: 15,
      output_tokens: 7,
      total_tokens: 22,
      input_token_details: { cache_read: 5, cache_creation: 2 },
    });
  });

  it("honors an external abort signal and cancels the run", async () => {
    const controller = new AbortController();
    stubAgent(
      stubRun({ result: "never" }, () => {
        controller.abort(new Error("caller went away"));
        return new Promise<never>(() => {});
      }),
    );

    const llm = new ChatCursor({ model: "composer-2.5" });

    await expect(
      llm.invoke(MESSAGES, { signal: controller.signal }),
    ).rejects.toThrow(/caller went away/);
    expect(cancelMock).toHaveBeenCalled();
    expect(closeMock).toHaveBeenCalled();
  });
});

describe("ChatCursor scratch cleanup", () => {
  it("removes its scratch workspace on dispose", async () => {
    const { existsSync } = await import("node:fs");
    stubAgent(stubRun({ result: "ok" }));

    const llm = new ChatCursor({ model: "composer-2.5" });
    await llm.invoke(MESSAGES);

    const cwd = createMock.mock.calls[0]?.[0].local.cwd as string;
    expect(existsSync(cwd)).toBe(true);

    await llm.dispose();
    expect(existsSync(cwd)).toBe(false);
  });
});
