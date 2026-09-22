import type { ToolDefinition } from "@langchain/core/language_models/base";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { describe, expect, it } from "vitest";
import {
  buildToolContract,
  CHAT_GUARDRAILS,
  extractJsonObject,
  parseToolCalls,
  partitionSystemMessages,
  serializeMessages,
} from "../src/ChatCursorMessages.js";

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

describe("serializeMessages", () => {
  it("wraps messages in role tags in order", () => {
    const { text, images } = serializeMessages([
      new SystemMessage("You are a planner."),
      new HumanMessage("Plan the goal."),
    ]);

    expect(text).toBe(
      "<system>\nYou are a planner.\n</system>\n\n<user>\nPlan the goal.\n</user>",
    );
    expect(images).toEqual([]);
  });

  it("maps ai messages to assistant tags", () => {
    const { text } = serializeMessages([
      new HumanMessage("Hi"),
      new AIMessage("Hello!"),
    ]);

    expect(text).toContain("<assistant>\nHello!\n</assistant>");
  });

  it("joins text blocks and extracts data-URL images", () => {
    const { text, images } = serializeMessages([
      new HumanMessage({
        content: [
          { type: "text", text: "What is on the screenshot?" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,aGVsbG8=" },
          },
        ],
      }),
    ]);

    expect(text).toContain("What is on the screenshot?");
    expect(text).toContain("[image 1 attached]");
    expect(images).toEqual([{ data: "aGVsbG8=", mimeType: "image/png" }]);
  });

  it("keeps non-data image URLs in the text", () => {
    const { text, images } = serializeMessages([
      new HumanMessage({
        content: [
          { type: "image_url", image_url: { url: "https://foo.bar/img.png" } },
        ],
      }),
    ]);

    expect(text).toContain("https://foo.bar/img.png");
    expect(images).toEqual([]);
  });
});

describe("CHAT_GUARDRAILS", () => {
  it("forbids using the agent's own tools", () => {
    expect(CHAT_GUARDRAILS).toMatch(/do not/i);
    expect(CHAT_GUARDRAILS).toMatch(/tools/i);
  });
});

describe("buildToolContract", () => {
  it("embeds the tool schemas and the response shape", () => {
    const contract = buildToolContract([CLICK_TOOL]);

    expect(contract).toContain('"click"');
    expect(contract).toContain('"tool_calls"');
    expect(contract).toContain("Click an element");
  });
});

describe("extractJsonObject", () => {
  it("returns a bare JSON object as-is", () => {
    expect(extractJsonObject('{"a": 1}')).toBe('{"a": 1}');
  });

  it("extracts JSON from a fenced code block", () => {
    const text = 'Here you go:\n```json\n{"a": {"b": 2}}\n```\nDone.';
    expect(extractJsonObject(text)).toBe('{"a": {"b": 2}}');
  });

  it("extracts JSON surrounded by prose", () => {
    expect(extractJsonObject('Sure! {"a": 1} Hope that helps.')).toBe(
      '{"a": 1}',
    );
  });

  it("handles braces inside strings", () => {
    const json = '{"a": "curly } brace {", "b": 1}';
    expect(extractJsonObject(`prefix ${json} suffix`)).toBe(json);
  });

  it("returns null when there is no JSON object", () => {
    expect(extractJsonObject("no json here")).toBeNull();
    expect(extractJsonObject("{unbalanced")).toBeNull();
  });
});

describe("parseToolCalls", () => {
  it("parses tool calls with arguments", () => {
    const calls = parseToolCalls(
      '{"tool_calls": [{"name": "click", "arguments": {"id": 5}}]}',
    );

    expect(calls).toEqual([{ name: "click", args: { id: 5 } }]);
  });

  it("accepts the args key as an alias for arguments", () => {
    const calls = parseToolCalls(
      '{"tool_calls": [{"name": "click", "args": {"id": 5}}]}',
    );

    expect(calls).toEqual([{ name: "click", args: { id: 5 } }]);
  });

  it("defaults missing arguments to an empty object", () => {
    const calls = parseToolCalls('{"tool_calls": [{"name": "noop"}]}');

    expect(calls).toEqual([{ name: "noop", args: {} }]);
  });

  it("parses multiple tool calls in order", () => {
    const calls = parseToolCalls(
      '{"tool_calls": [{"name": "click", "arguments": {"id": 1}}, {"name": "type", "arguments": {"id": 2, "text": "hi"}}]}',
    );

    expect(calls.map((call) => call.name)).toEqual(["click", "type"]);
  });

  it("parses an empty tool call list", () => {
    expect(parseToolCalls('{"tool_calls": []}')).toEqual([]);
  });

  it("parses tool calls wrapped in a fenced block with prose", () => {
    const calls = parseToolCalls(
      'Let me click it.\n```json\n{"tool_calls": [{"name": "click", "arguments": {"id": 9}}]}\n```',
    );

    expect(calls).toEqual([{ name: "click", args: { id: 9 } }]);
  });

  it("throws on text without JSON", () => {
    expect(() => parseToolCalls("I clicked the button for you.")).toThrow(
      /JSON/,
    );
  });

  it("throws on JSON with the wrong shape", () => {
    expect(() => parseToolCalls('{"actions": ["click"]}')).toThrow();
  });
});

describe("partitionSystemMessages", () => {
  it("lifts system messages out of the history", () => {
    const { systemText, rest } = partitionSystemMessages([
      new SystemMessage("Be terse."),
      new HumanMessage("Hi"),
    ]);

    expect(systemText).toBe("Be terse.");
    expect(rest).toHaveLength(1);
    expect(rest[0]?.getType()).toBe("human");
  });

  it("joins several system messages in order", () => {
    const { systemText } = partitionSystemMessages([
      new SystemMessage("Be terse."),
      new HumanMessage("Hi"),
      new SystemMessage("Answer in French."),
    ]);

    expect(systemText).toBe("Be terse.\n\nAnswer in French.");
  });

  it("leaves the history untouched when there is nothing to lift", () => {
    const messages = [new HumanMessage("Hi"), new AIMessage("Hello")];
    const { systemText, rest } = partitionSystemMessages(messages);

    expect(systemText).toBe("");
    expect(rest).toBe(messages);
  });

  it("does not lift whitespace-only system messages", () => {
    // The SDK rejects an empty `systemPrompt`, so these have to stay inline.
    const messages = [new SystemMessage("   "), new HumanMessage("Hi")];
    const { systemText, rest } = partitionSystemMessages(messages);

    expect(systemText).toBe("");
    expect(rest).toBe(messages);
  });

  it("does not lift system messages carrying images", () => {
    // The attachment rides with the user payload, so splitting the text away
    // from its `[image N attached]` placeholder would scramble the numbering.
    const messages = [
      new SystemMessage({
        content: [
          { type: "text", text: "Match this style" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,aGVsbG8=" },
          },
        ],
      }),
      new HumanMessage("Hi"),
    ];
    const { systemText, rest } = partitionSystemMessages(messages);

    expect(systemText).toBe("");
    expect(rest).toBe(messages);
  });
});

describe("serializeMessages role tags", () => {
  it("tags tool results so the model can tell them from user turns", () => {
    const { text } = serializeMessages([
      new HumanMessage("Click it"),
      new ToolMessage({ content: "clicked", tool_call_id: "call_1" }),
    ]);

    expect(text).toContain("<tool_result>\nclicked\n</tool_result>");
  });
});
