import type {
  AgentOptions,
  CursorAgentPlatform,
  LocalAgentStore,
  SDKAgent,
} from "@cursor/sdk";
import fs from "node:fs/promises";
import os from "node:os";
import { join } from "node:path";

export type CursorSdkModule = typeof import("@cursor/sdk");

/** How long an idle instance holds the warm Cursor executor. */
export const DEFAULT_KEEP_ALIVE_MS = 60_000;

/**
 * The parts of an agent's configuration that `ChatCursor` decides per call.
 * `CursorRuntime` fills in the scratch workspace and the run store.
 */
export interface AgentRequest {
  apiKey?: string | undefined;
  model: { id: string };
  sandbox: boolean;
  /** Built-in tools the agent may use. `[]` makes it text-only. */
  tools?: string[] | undefined;
  systemPrompt?: string | undefined;
}

/**
 * Owns the process-level Cursor state that outlives a single generation: the
 * scratch dirs, the run store, the agent platform, and the prewarmed local
 * executor.
 *
 * The executor is the expensive part of a local agent — auth, feature gates,
 * the dashboard client, sandbox policy, and a workspace scan — and the SDK
 * reference-counts it, tearing it down when the last agent closes. Holding a
 * `prewarmLocalWorkspace` lease keeps that count above zero between calls so
 * each generation only pays for the model, while still creating (and closing)
 * a throwaway agent per call so no conversation state leaks across
 * invocations.
 */
export class CursorRuntime {
  #sdkLoader?: (() => Promise<CursorSdkModule>) | undefined;
  #keepAliveMs: number;

  #sdk?: Promise<CursorSdkModule>;
  #scratchDirs?:
    | Promise<{ root: string; workspaceDir: string; storeDir: string }>
    | undefined;
  #store?: Promise<LocalAgentStore>;
  #platform?: Promise<CursorAgentPlatform | undefined>;

  #warm?: { key: string; release: () => Promise<void> };
  #warming?: Promise<void>;
  #idleTimer?: ReturnType<typeof setTimeout>;
  #disposed = false;

  constructor(fields?: {
    sdkLoader?: (() => Promise<CursorSdkModule>) | undefined;
    keepAliveMs?: number | undefined;
  }) {
    this.#sdkLoader = fields?.sdkLoader;
    this.#keepAliveMs = fields?.keepAliveMs ?? DEFAULT_KEEP_ALIVE_MS;
  }

  /**
   * Whether the loaded SDK understands the agent options that replace this
   * package's prompt-level workarounds (`tools`, `systemPrompt`) and can keep
   * an executor warm. Hosts may supply their own module through `sdkLoader`,
   * so this cannot be assumed from the declared dependency range.
   */
  async supportsNativeOptions(): Promise<boolean> {
    return (await this.#getPlatform()) !== undefined;
  }

  async createAgent(request: AgentRequest): Promise<SDKAgent> {
    const platform = await this.#getPlatform();
    if (platform) return platform.createAgent(await this.#toOptions(request));

    // Older SDK: no platform to hold the store, so it rides on each agent.
    const sdk = await this.#importSdk();
    return sdk.Agent.create(await this.#toOptions(request, true));
  }

  /** Restart the idle countdown without touching the lease itself. */
  touch(): void {
    if (this.#warm) this.#armIdleTimer();
  }

  /**
   * Hold the local executor warm, building it if this is the first call.
   *
   * Start this alongside {@link createAgent} rather than awaiting it: the
   * executor build and the agent's model-catalog lookup hit different things,
   * and overlapping them takes roughly a third off a cold start. On a warm
   * instance it is a no-op that just restarts the idle countdown.
   *
   * Prewarming is a pure optimization, so failures are swallowed — the next
   * `send` rebuilds.
   */
  async ensureWarm(request: AgentRequest): Promise<void> {
    if (this.#keepAliveMs <= 0 || this.#disposed) return;

    const key = this.#warmKey(request);
    if (this.#warm?.key !== key) {
      // Serialize against any warm already in flight so concurrent calls
      // cannot acquire two leases and leak one.
      const warming = (this.#warming ?? Promise.resolve())
        .catch(() => {})
        .then(() => this.#acquireWarm(key, request));
      this.#warming = warming;
      await warming.catch(() => {});
    }

    this.#armIdleTimer();
  }

  async dispose(): Promise<void> {
    this.#disposed = true;
    await this.#warming?.catch(() => {});
    await this.#releaseWarm();
    await this.#removeScratchDirs();
  }

  async #acquireWarm(key: string, request: AgentRequest): Promise<void> {
    if (this.#disposed || this.#warm?.key === key) return;

    const platform = await this.#getPlatform();
    if (!platform) return;

    const release = await platform.prewarmLocalWorkspace(
      await this.#toOptions(request),
    );

    // Acquire before releasing the stale lease: when only `sandboxOptions`
    // changed these are different executors, but dropping to zero references
    // first would let the SDK tear down one we are about to want.
    if (this.#disposed || this.#warm?.key === key) {
      await release().catch(() => {});
      return;
    }
    const previous = this.#warm;
    this.#warm = { key, release };
    if (previous) await previous.release().catch(() => {});
  }

  async #releaseWarm(): Promise<void> {
    if (this.#idleTimer) {
      clearTimeout(this.#idleTimer);
      this.#idleTimer = undefined;
    }
    const warm = this.#warm;
    this.#warm = undefined;
    if (warm) await warm.release().catch(() => {});
  }

  #armIdleTimer(): void {
    if (this.#idleTimer) clearTimeout(this.#idleTimer);
    this.#idleTimer = undefined;
    if (!this.#warm || this.#keepAliveMs <= 0) return;

    const timer = setTimeout(() => {
      void this.#releaseWarm();
    }, this.#keepAliveMs);
    // Never hold the event loop open: a script that calls the model once and
    // never disposes the instance must still exit on its own.
    timer.unref?.();
    this.#idleTimer = timer;
  }

  /**
   * The executor cache key, as the SDK derives it, restricted to the fields
   * this package varies. Everything else it hashes (the scratch cwd, empty
   * setting sources, no MCP servers) is fixed for the life of the instance.
   */
  #warmKey(request: AgentRequest): string {
    return `${request.sandbox ? "sandboxed" : "unsandboxed"}:${request.apiKey ?? ""}`;
  }

  async #toOptions(
    request: AgentRequest,
    includeStore = false,
  ): Promise<AgentOptions> {
    const { workspaceDir } = await this.#ensureScratchDirs();

    // The empty workspace, disabled setting sources, and sandbox (where the
    // environment supports it) keep the throwaway agent away from the user's
    // files and MCP servers; the store keeps its run records out of the
    // user's Cursor state root.
    const local: NonNullable<AgentOptions["local"]> = {
      cwd: workspaceDir,
      settingSources: [],
      sandboxOptions: { enabled: request.sandbox },
    };
    if (includeStore) local.store = await this.#getStore();

    return {
      ...(request.apiKey ? { apiKey: request.apiKey } : {}),
      model: request.model,
      ...(request.tools ? { tools: request.tools } : {}),
      ...(request.systemPrompt ? { systemPrompt: request.systemPrompt } : {}),
      local,
    };
  }

  #getPlatform(): Promise<CursorAgentPlatform | undefined> {
    if (!this.#platform) {
      const pending = (async () => {
        const sdk = await this.#importSdk();

        // `sdkLoader` hosts may supply a module older than the one this
        // package depends on. Such a module ignores `tools` / `systemPrompt`
        // silently, so the caller falls back to the prompt-level guardrails.
        if (typeof sdk.createAgentPlatform !== "function") return undefined;

        const platform = await sdk.createAgentPlatform({
          localStore: await this.#getStore(),
        });
        if (typeof platform?.prewarmLocalWorkspace !== "function") {
          return undefined;
        }
        return platform;
      })();

      this.#platform = pending;
      pending.catch(() => {
        if (this.#platform === pending) this.#platform = undefined;
      });
    }
    return this.#platform;
  }

  #getStore(): Promise<LocalAgentStore> {
    this.#store ??= (async () => {
      const sdk = await this.#importSdk();
      const { storeDir } = await this.#ensureScratchDirs();
      return new sdk.JsonlLocalAgentStore(storeDir);
    })();
    return this.#store;
  }

  #ensureScratchDirs() {
    this.#scratchDirs ??= (async () => {
      const root = await fs.mkdtemp(join(os.tmpdir(), "langchain-cursor-"));
      const workspaceDir = join(root, "workspace");
      const storeDir = join(root, "store");
      await fs.mkdir(workspaceDir, { recursive: true });
      await fs.mkdir(storeDir, { recursive: true });
      return { root, workspaceDir, storeDir };
    })();
    return this.#scratchDirs;
  }

  /**
   * Remove the scratch workspace and run store. Only safe once the lease is
   * released and no agent is in flight, so `dispose` is the only caller.
   */
  async #removeScratchDirs(): Promise<void> {
    const dirs = this.#scratchDirs;
    if (!dirs) return;
    this.#scratchDirs = undefined;
    this.#store = undefined;
    this.#platform = undefined;
    try {
      const { root } = await dirs;
      await fs.rm(root, { recursive: true, force: true });
    } catch {
      // A leftover temp dir is not worth failing a dispose over.
    }
  }

  #importSdk(): Promise<CursorSdkModule> {
    this.#sdk ??= (this.#sdkLoader ? this.#sdkLoader() : loadCursorSdk()).catch(
      (error: unknown) => {
        this.#sdk = undefined;
        throw error;
      },
    );
    return this.#sdk;
  }
}

async function loadCursorSdk(): Promise<CursorSdkModule> {
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
