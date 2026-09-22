import type {
  RunResult,
  SDKAgent,
  SDKUserMessage,
  TokenUsage,
} from "@cursor/sdk";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type {
  BaseLanguageModelInput,
  StructuredOutputMethodOptions,
  ToolDefinition,
} from "@langchain/core/language_models/base";
import {
  BaseChatModel,
  type BaseChatModelCallOptions,
  type BaseChatModelParams,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import {
  assembleStructuredOutputPipeline,
  createFunctionCallingParser,
} from "@langchain/core/language_models/structured_output";
import {
  AIMessage,
  type AIMessageFields,
  type BaseMessage,
  type StandardMessageStructure,
  type UsageMetadata,
} from "@langchain/core/messages";
import type { ToolCall } from "@langchain/core/messages/tool";
import type { ChatResult } from "@langchain/core/outputs";
import type { Runnable } from "@langchain/core/runnables";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { toJsonSchema } from "@langchain/core/utils/json_schema";
import type { SerializableSchema } from "@langchain/core/utils/standard_schema";
import type { ZodV3Like, ZodV4Like } from "@langchain/core/utils/types";
import {
  buildToolContract,
  CHAT_GUARDRAILS,
  type CursorToolCall,
  parseToolCalls,
  partitionSystemMessages,
  serializeMessages,
} from "./ChatCursorMessages.js";
import {
  type AgentRequest,
  CursorRuntime,
  type CursorSdkModule,
} from "./CursorRuntime.js";

export type { CursorSdkModule } from "./CursorRuntime.js";

export interface ChatCursorCallOptions extends BaseChatModelCallOptions {
  tools?: ToolDefinition[];
}

export interface ChatCursorFields extends BaseChatModelParams {
  model?: string;
  apiKey?: string;
  /**
   * Overrides how `@cursor/sdk` is loaded. Hosts whose runtime cannot
   * resolve node_modules (e.g. single-file compiled binaries) can supply a
   * loader that provisions the SDK themselves. Loader errors surface
   * unwrapped.
   */
  sdkLoader?: () => Promise<CursorSdkModule>;
  /**
   * How long, in milliseconds, an idle instance keeps the Cursor local
   * executor warm so later calls skip its startup cost. Defaults to 60s; set
   * `0` to tear it down after every call. The timer never holds the process
   * open — call {@link ChatCursor.dispose} to release it immediately.
   */
  keepAliveMs?: number;
}

const DEFAULT_MODEL = "composer-2.5";

/**
 * LangChain chat model backed by the Cursor Agents SDK local runtime.
 *
 * Cursor exposes no chat-completions API, so every `_generate` call runs a
 * fresh throwaway local agent against an empty scratch workspace: the message
 * history is serialized into a single prompt, system messages become the
 * agent's own `systemPrompt`, and the agent is given no built-in tools so it
 * can only answer with text. Tool calling is emulated on top of that through
 * a prompt contract whose JSON reply is parsed back into LangChain
 * `tool_calls`.
 *
 * Agents are throwaway, but the local executor behind them is not: it is kept
 * warm between calls so only the first one pays its startup cost. See
 * {@link ChatCursorFields.keepAliveMs} and {@link ChatCursor.dispose}.
 */
export class ChatCursor extends BaseChatModel<ChatCursorCallOptions> {
  model: string;
  apiKey?: string | undefined;

  #runtime: CursorRuntime;
  #sandbox = true;
  #systemPrompt = true;

  static override lc_name(): string {
    return "ChatCursor";
  }

  constructor(fields?: ChatCursorFields) {
    super(fields ?? {});
    this.model = fields?.model ?? DEFAULT_MODEL;
    this.apiKey = fields?.apiKey ?? process.env.CURSOR_API_KEY;
    this.#runtime = new CursorRuntime({
      sdkLoader: fields?.sdkLoader,
      keepAliveMs: fields?.keepAliveMs,
    });
  }

  /**
   * Release the warm Cursor executor this instance is holding. Optional —
   * an idle instance releases it on its own and never blocks process exit —
   * but a long-lived host should call it when the model is no longer needed.
   */
  async dispose(): Promise<void> {
    await this.#runtime.dispose();
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.dispose();
  }

  _llmType(): string {
    return "cursor";
  }

  override invocationParams(options?: this["ParsedCallOptions"]) {
    return { model: this.model, tools: options?.tools };
  }

  override bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<ChatCursorCallOptions>,
  ) {
    return this.withConfig({
      tools: tools.map((tool) => convertToOpenAITool(tool) as ToolDefinition),
      ...kwargs,
    });
  }

  // NOTE: The base implementation's output parser rejects anything that is
  // not an AIMessageChunk. Cache hits deserialize into plain AIMessages, so
  // this override assembles the same pipeline with a chunk-agnostic parser.
  override withStructuredOutput<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Mirrors the base class signature
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | SerializableSchema<RunOutput>
      | ZodV4Like<RunOutput>
      | ZodV3Like<RunOutput>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Mirrors the base class signature
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<false>,
  ): Runnable<BaseLanguageModelInput, RunOutput>;
  override withStructuredOutput<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Mirrors the base class signature
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | SerializableSchema<RunOutput>
      | ZodV4Like<RunOutput>
      | ZodV3Like<RunOutput>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Mirrors the base class signature
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<true>,
  ): Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }>;
  override withStructuredOutput<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Mirrors the base class signature
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | SerializableSchema<RunOutput>
      | ZodV4Like<RunOutput>
      | ZodV3Like<RunOutput>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Mirrors the base class signature
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<boolean>,
  ):
    | Runnable<BaseLanguageModelInput, RunOutput>
    | Runnable<
        BaseLanguageModelInput,
        { raw: BaseMessage; parsed: RunOutput }
      > {
    const name = config?.name ?? "extract";
    const asJsonSchema = toJsonSchema(outputSchema as never) as Record<
      string,
      unknown
    >;
    const description =
      typeof asJsonSchema.description === "string"
        ? asJsonSchema.description
        : "A function available to call.";

    const tools: ToolDefinition[] = [
      {
        type: "function",
        function: { name, description, parameters: asJsonSchema },
      },
    ];
    const parser = createFunctionCallingParser(
      outputSchema as Record<string, unknown>,
      name,
    );

    return assembleStructuredOutputPipeline(
      this.bindTools(tools),
      parser,
      config?.includeRaw,
      config?.includeRaw ? "StructuredOutputRunnable" : "StructuredOutput",
    ) as Runnable<BaseLanguageModelInput, never>;
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    // Both degradations below flip a sticky flag, so this retries at most
    // once per flag and then gives up.
    let speculative: unknown;
    for (;;) {
      try {
        return await this.#generateOnce(messages, options);
      } catch (error) {
        const degraded = this.#degrade(error);
        if (!degraded) throw speculative ?? error;
        // The system-prompt gate is only recognizable by its message text, so
        // a match may be a false positive. Keep the original error to report
        // if the retry fails too — it is the more trustworthy one.
        if (degraded === "system-prompt") speculative ??= error;
      }
    }
  }

  /**
   * Turn off a capability the environment or account does not support and
   * report whether the call is worth retrying.
   */
  #degrade(error: unknown): "sandbox" | "system-prompt" | undefined {
    // Cursor SDK sandboxing is unavailable in some environments (e.g. WSL)
    // and the SDK hard-fails when it is requested there. The empty scratch
    // workspace, disabled setting sources, and the text-only toolset still
    // apply, so degrade gracefully and stop requesting the sandbox.
    if (this.#sandbox && isSandboxUnsupportedError(error)) {
      console.warn(
        "[langchain-cursor] Cursor SDK sandboxing is not supported in this environment, retrying without it.",
      );
      this.#sandbox = false;
      return "sandbox";
    }

    // `systemPrompt` is gated server-side and rejected on the first send
    // rather than at create, so fall back to inlining system messages in the
    // prompt the way this package did before the option existed.
    if (this.#systemPrompt && isSystemPromptUnsupportedError(error)) {
      console.warn(
        "[langchain-cursor] Cursor rejected a custom system prompt for this account, falling back to inline system messages.",
      );
      this.#systemPrompt = false;
      return "system-prompt";
    }

    return undefined;
  }

  async #generateOnce(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
  ): Promise<ChatResult> {
    // An SDK old enough to lack the platform API also ignores `tools` and
    // `systemPrompt`, so fall back to stating both in the prompt.
    const native = await this.#runtime.supportsNativeOptions();
    const tools = options.tools ?? [];

    let systemPrompt: string | undefined;
    let history = messages;
    if (native && this.#systemPrompt) {
      const partitioned = partitionSystemMessages(messages);
      if (partitioned.systemText) {
        systemPrompt = partitioned.systemText;
        history = partitioned.rest;
      }
    }

    const prompt = serializeMessages(history);
    const sections = [prompt.text];
    if (!native) sections.push(CHAT_GUARDRAILS);
    if (tools.length) sections.push(buildToolContract(tools));
    const text = sections.join("\n\n");

    const request: AgentRequest = {
      ...(this.apiKey ? { apiKey: this.apiKey } : {}),
      model: { id: this.model },
      sandbox: this.#sandbox,
      // No built-in tools: the agent is used as a chat model, and a
      // text-only reply is exactly what the tool contract asks it to produce.
      ...(native ? { tools: [] } : {}),
      ...(systemPrompt ? { systemPrompt } : {}),
    };

    // Build the executor and resolve the agent at the same time. They hit
    // different things — a workspace/auth bootstrap versus the model catalog
    // — so overlapping them takes roughly a third off a cold start. If the
    // sandbox then turns out to be unsupported, this warmed the wrong key and
    // the retry rebuilds; that costs one executor build, once per instance.
    const warming = this.#runtime.ensureWarm(request).catch(() => {});
    const agent = await this.#runtime.createAgent(request);
    try {
      const first = await this.#send(
        agent,
        { text, images: prompt.images },
        options,
      );

      // NOTE: The raw reply is kept as the message content even when tool
      // calls are parsed out of it — callers (e.g. caching layers) may rely
      // on generations having non-empty content.
      let content = first.result ?? "";
      let usage = toUsageMetadata(first.usage);
      let toolCalls: ToolCall[] = [];

      if (tools.length) {
        try {
          toolCalls = toLchainToolCalls(parseToolCalls(content));
        } catch (error) {
          const retry = await this.#send(
            agent,
            { text: correctivePrompt(error) },
            options,
          );
          content = retry.result ?? "";
          toolCalls = toLchainToolCalls(parseToolCalls(content));
          usage = mergeUsage(usage, toUsageMetadata(retry.usage));
        }
      }

      const fields: AIMessageFields<StandardMessageStructure> = {
        id: first.id,
        content,
        tool_calls: toolCalls,
        response_metadata: { model_name: this.model },
      };
      if (usage) fields.usage_metadata = usage;
      const message = new AIMessage(fields);

      return { generations: [{ text: content, message }] };
    } finally {
      // Settle the lease before dropping this agent's reference: if the
      // prewarm were still in flight the count could reach zero here and the
      // SDK would tear down the executor we are paying to keep.
      await warming;
      agent.close();
      this.#runtime.touch();
    }
  }

  async #send(
    agent: SDKAgent,
    payload: SDKUserMessage,
    options: this["ParsedCallOptions"],
  ): Promise<RunResult> {
    const signals: AbortSignal[] = [];
    if (options.signal) signals.push(options.signal);
    if (options.timeout) signals.push(AbortSignal.timeout(options.timeout));
    const signal = signals.length ? AbortSignal.any(signals) : undefined;

    try {
      const run = await agent.send(payload);

      let result: RunResult;
      if (signal) {
        const aborted = new Promise<never>((_, reject) => {
          const onAbort = () =>
            reject(
              signal.reason instanceof Error
                ? signal.reason
                : new Error(String(signal.reason ?? "Aborted")),
            );
          if (signal.aborted) onAbort();
          else signal.addEventListener("abort", onAbort, { once: true });
        });

        try {
          result = await Promise.race([run.wait(), aborted]);
        } catch (error) {
          run.cancel().catch(() => {});
          throw error;
        }
      } else {
        result = await run.wait();
      }

      if (result.status !== "finished") {
        throw new Error(
          `Cursor agent run ${result.status}: ${result.error?.message ?? "no error details"}`,
        );
      }

      return result;
    } catch (error) {
      throw mapSdkError(error);
    }
  }

}

function correctivePrompt(error: unknown): string {
  const detail = error instanceof Error ? error.message : String(error);
  return (
    `Your previous reply was invalid: ${detail}\n` +
    'Respond again with ONLY a JSON object of the form {"tool_calls": ' +
    '[{"name": "<tool_name>", "arguments": {<parameters>}}]} and no other text.'
  );
}

function toLchainToolCalls(calls: CursorToolCall[]): ToolCall[] {
  return calls.map((call) => ({
    type: "tool_call" as const,
    id: `call_${crypto.randomUUID()}`,
    name: call.name,
    args: call.args,
  }));
}

function toUsageMetadata(usage?: TokenUsage): UsageMetadata | undefined {
  if (!usage) return undefined;
  return {
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens,
    total_tokens: usage.totalTokens,
    input_token_details: {
      cache_read: usage.cacheReadTokens,
      cache_creation: usage.cacheWriteTokens,
    },
  };
}

function mergeUsage(
  a: UsageMetadata | undefined,
  b: UsageMetadata | undefined,
): UsageMetadata | undefined {
  if (!a || !b) return a ?? b;
  return {
    input_tokens: a.input_tokens + b.input_tokens,
    output_tokens: a.output_tokens + b.output_tokens,
    total_tokens: a.total_tokens + b.total_tokens,
    input_token_details: {
      cache_read:
        (a.input_token_details?.cache_read ?? 0) +
        (b.input_token_details?.cache_read ?? 0),
      cache_creation:
        (a.input_token_details?.cache_creation ?? 0) +
        (b.input_token_details?.cache_creation ?? 0),
    },
  };
}

function isSandboxUnsupportedError(error: unknown): boolean {
  return (
    error instanceof Error &&
    error.name === "ConfigurationError" &&
    /sandbox/i.test(error.message)
  );
}

/**
 * The `systemPrompt` gate has no dedicated error class or code — the backend
 * rejects the run with an invalid-argument error whose message names the
 * `--system-prompt` flag. Match that, plus the looser wording, and rely on
 * the caller to treat a match as speculative.
 */
function isSystemPromptUnsupportedError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  if (/--system-prompt/.test(error.message)) return true;
  return (
    /system[_ -]?prompt/i.test(error.message) &&
    /invalid|not (enabled|supported|allowed)|permission|access/i.test(
      error.message,
    )
  );
}

function mapSdkError(error: unknown): Error {
  if (!(error instanceof Error)) return new Error(String(error));

  // The SDK's RateLimitError name and AbortSignal.timeout's TimeoutError name
  // both already match common retry wrappers' detection.
  if (error.name === "RateLimitError" || error.name === "TimeoutError") {
    return error;
  }

  // The SDK flags transient failures (e.g. NetworkError on 503/504) as
  // retryable; surface them as TimeoutError so retry wrappers pick them up.
  if (
    "isRetryable" in error &&
    (error as { isRetryable?: boolean }).isRetryable === true
  ) {
    const wrapped = new Error(error.message, { cause: error });
    wrapped.name = "TimeoutError";
    return wrapped;
  }

  return error;
}
