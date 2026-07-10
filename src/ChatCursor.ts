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
import fs from "node:fs/promises";
import os from "node:os";
import { join } from "node:path";
import {
  buildToolContract,
  CHAT_GUARDRAILS,
  type CursorToolCall,
  parseToolCalls,
  serializeMessages,
} from "./ChatCursorMessages.js";

export interface ChatCursorCallOptions extends BaseChatModelCallOptions {
  tools?: ToolDefinition[];
}

export type CursorSdkModule = typeof import("@cursor/sdk");

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
}

const DEFAULT_MODEL = "composer-2.5";

/**
 * LangChain chat model backed by the Cursor Agents SDK local runtime.
 *
 * Cursor exposes no chat-completions API, so every `_generate` call runs a
 * fresh throwaway local agent: the message history is serialized into a
 * single prompt, the agent is sandboxed away from its own harness tools, and
 * tool calling is emulated through a prompt contract whose JSON reply is
 * parsed back into LangChain `tool_calls`.
 */
export class ChatCursor extends BaseChatModel<ChatCursorCallOptions> {
  model: string;
  apiKey?: string | undefined;

  #scratchDirs?: Promise<{ workspaceDir: string; storeDir: string }>;
  #sandbox = true;
  #sdkLoader?: (() => Promise<CursorSdkModule>) | undefined;

  static override lc_name(): string {
    return "ChatCursor";
  }

  constructor(fields?: ChatCursorFields) {
    super(fields ?? {});
    this.model = fields?.model ?? DEFAULT_MODEL;
    this.apiKey = fields?.apiKey ?? process.env.CURSOR_API_KEY;
    this.#sdkLoader = fields?.sdkLoader;
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
    try {
      return await this.#generateOnce(messages, options);
    } catch (error) {
      // Cursor SDK sandboxing is unavailable in some environments (e.g. WSL)
      // and the SDK hard-fails when it is requested there. The empty scratch
      // workspace, disabled setting sources, and prompt guardrails still
      // apply, so degrade gracefully and stop requesting the sandbox.
      if (!this.#sandbox || !isSandboxUnsupportedError(error)) throw error;

      console.warn(
        "[langchain-cursor] Cursor SDK sandboxing is not supported in this environment, retrying without it.",
      );
      this.#sandbox = false;
      return await this.#generateOnce(messages, options);
    }
  }

  async #generateOnce(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
  ): Promise<ChatResult> {
    const prompt = serializeMessages(messages);
    const tools = options.tools ?? [];

    const sections = [prompt.text, CHAT_GUARDRAILS];
    if (tools.length) sections.push(buildToolContract(tools));
    const text = sections.join("\n\n");

    const agent = await this.#createAgent();
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
      agent.close();
    }
  }

  async #createAgent(): Promise<SDKAgent> {
    const { Agent, JsonlLocalAgentStore } = await this.#importCursorSdk();
    const { workspaceDir, storeDir } = await this.#ensureScratchDirs();

    // The empty workspace, disabled setting sources, and sandbox (where the
    // environment supports it) all keep the throwaway agent from touching the
    // user's files or MCP servers; the store keeps its run records out of the
    // user's Cursor state root.
    return Agent.create({
      ...(this.apiKey ? { apiKey: this.apiKey } : {}),
      model: { id: this.model },
      local: {
        cwd: workspaceDir,
        settingSources: [],
        sandboxOptions: { enabled: this.#sandbox },
        store: new JsonlLocalAgentStore(storeDir),
      },
    });
  }

  #ensureScratchDirs() {
    this.#scratchDirs ??= (async () => {
      const root = await fs.mkdtemp(join(os.tmpdir(), "langchain-cursor-"));
      const workspaceDir = join(root, "workspace");
      const storeDir = join(root, "store");
      await fs.mkdir(workspaceDir, { recursive: true });
      await fs.mkdir(storeDir, { recursive: true });
      return { workspaceDir, storeDir };
    })();
    return this.#scratchDirs;
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

  async #importCursorSdk(): Promise<CursorSdkModule> {
    if (this.#sdkLoader) return this.#sdkLoader();

    try {
      return await import("@cursor/sdk");
    } catch (error) {
      // @cursor/sdk ships a webpack-chunked dist that loads chunks dynamically
      // (`require("./" + chunkId + ".js")`), so it cannot be inlined by
      // bundlers and must be resolvable from node_modules at runtime.
      throw new Error(
        "Failed to load @cursor/sdk — langchain-cursor requires a runtime " +
          "that resolves node_modules; single-file compiled binaries are not " +
          "supported.",
        { cause: error },
      );
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
