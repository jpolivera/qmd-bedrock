/**
 * bedrock-expand.ts - Optional AWS Bedrock query-expansion backend for QMD.
 *
 * Activated when QMD_EXPAND_BACKEND=bedrock. Routes query expansion through a
 * Bedrock chat model (default: amazon.nova-micro-v1:0 — same model the
 * fact-extraction pipeline already uses) instead of the local Qwen3-1.7B GGUF
 * which takes 20+s on CPU and causes MCP stdio timeouts on `qmd query`.
 *
 * Default model: amazon.nova-micro-v1:0
 * Override with QMD_EXPAND_MODEL.
 *
 * Region resolves from QMD_BEDROCK_REGION → AWS_REGION → us-east-1.
 *
 * Output contract: returns Queryable[] with type ∈ {lex,vec,hyde}, matching
 * the local expandQuery shape so callers see no behavioral difference.
 */

import type { Queryable } from "./llm.js";

const DEFAULT_BEDROCK_EXPAND_MODEL = "amazon.nova-micro-v1:0";
const DEFAULT_BEDROCK_REGION = "us-east-1";
const MAX_RETRIES = 3;
const BASE_BACKOFF_MS = 250;

export function isBedrockExpandEnabled(): boolean {
  return (process.env.QMD_EXPAND_BACKEND || "").trim().toLowerCase() === "bedrock";
}

export function bedrockExpandModel(): string {
  return process.env.QMD_EXPAND_MODEL?.trim() || DEFAULT_BEDROCK_EXPAND_MODEL;
}

function bedrockRegion(): string {
  return (process.env.QMD_BEDROCK_REGION || process.env.AWS_REGION || DEFAULT_BEDROCK_REGION).trim();
}

type BedrockClient = {
  send: (cmd: unknown) => Promise<{ body: Uint8Array }>;
};
let cachedClient: BedrockClient | null = null;
let CommandCtor: (new (input: { modelId: string; contentType: string; accept: string; body: Uint8Array }) => unknown) | null = null;

async function getClient(): Promise<{ client: BedrockClient; Cmd: NonNullable<typeof CommandCtor> }> {
  if (cachedClient && CommandCtor) return { client: cachedClient, Cmd: CommandCtor };
  let mod: typeof import("@aws-sdk/client-bedrock-runtime");
  try {
    mod = await import("@aws-sdk/client-bedrock-runtime");
  } catch (err) {
    throw new Error(
      `QMD_EXPAND_BACKEND=bedrock requires @aws-sdk/client-bedrock-runtime. ` +
        `Install it: npm i @aws-sdk/client-bedrock-runtime\n` +
        `(original error: ${(err as Error).message})`
    );
  }
  cachedClient = new mod.BedrockRuntimeClient({ region: bedrockRegion() }) as unknown as BedrockClient;
  CommandCtor = mod.InvokeModelCommand as unknown as NonNullable<typeof CommandCtor>;
  return { client: cachedClient, Cmd: CommandCtor };
}

const TRANSIENT_ERR = /^(Throttling|ServiceUnavailable|TimeoutError|RequestTimeout|InternalServer)/i;

const SYSTEM_PROMPT = `You expand search queries to improve recall. Output exactly 2-4 lines, each one of:
  lex: <keyword query for full-text search>
  vec: <semantic phrase for vector search>
  hyde: <hypothetical 1-2 sentence answer that would match relevant documents>

Rules:
- Output only the typed lines, no preamble or explanation.
- Each line must start with "lex: ", "vec: ", or "hyde: ".
- Always include at least one "vec:" line. Include "hyde:" only when a hypothetical answer would help.
- Stay focused on the user's query topic. Don't drift.`;

function buildPrompt(query: string, intent?: string, context?: string): string {
  let p = `Query: ${query}`;
  if (intent) p += `\nIntent: ${intent}`;
  if (context) p += `\nContext: ${context}`;
  return p;
}

interface NovaMessagesResponse {
  output?: { message?: { content?: Array<{ text?: string }> } };
}
interface AnthropicMessagesResponse {
  content?: Array<{ type?: string; text?: string }>;
}

function buildBody(modelId: string, query: string, intent?: string, context?: string): Record<string, unknown> {
  const userPrompt = buildPrompt(query, intent, context);
  // Nova family (amazon.nova-*) uses messages API with system/messages
  if (/^(?:us\.|eu\.|global\.)?amazon\.nova-/i.test(modelId)) {
    return {
      schemaVersion: "messages-v1",
      system: [{ text: SYSTEM_PROMPT }],
      messages: [{ role: "user", content: [{ text: userPrompt }] }],
      inferenceConfig: { maxTokens: 400, temperature: 0.4, topP: 0.9 },
    };
  }
  // Anthropic Claude on Bedrock
  if (/anthropic\.claude-/i.test(modelId)) {
    return {
      anthropic_version: "bedrock-2023-05-31",
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: userPrompt }],
      max_tokens: 400,
      temperature: 0.4,
    };
  }
  // Fall back to a generic prompt-style body
  return {
    inputText: `${SYSTEM_PROMPT}\n\n${userPrompt}`,
    textGenerationConfig: { maxTokenCount: 400, temperature: 0.4 },
  };
}

function extractText(modelId: string, json: unknown): string {
  if (/^(?:us\.|eu\.|global\.)?amazon\.nova-/i.test(modelId)) {
    const r = json as NovaMessagesResponse;
    return r.output?.message?.content?.[0]?.text ?? "";
  }
  if (/anthropic\.claude-/i.test(modelId)) {
    const r = json as AnthropicMessagesResponse;
    const blocks = r.content ?? [];
    return blocks.filter((b) => b.type === "text" || !b.type).map((b) => b.text ?? "").join("");
  }
  // Generic fallback: try common shapes
  const j = json as Record<string, unknown>;
  return (
    (j.completion as string) ||
    (j.outputText as string) ||
    JSON.stringify(j).slice(0, 200)
  );
}

async function invokeOnce(modelId: string, query: string, intent?: string, context?: string): Promise<string> {
  const { client, Cmd } = await getClient();
  const cmd = new Cmd({
    modelId,
    contentType: "application/json",
    accept: "application/json",
    body: new TextEncoder().encode(JSON.stringify(buildBody(modelId, query, intent, context))),
  });
  const res = await client.send(cmd);
  const json = JSON.parse(new TextDecoder().decode(res.body));
  return extractText(modelId, json);
}

async function expandWithRetry(modelId: string, query: string, intent?: string, context?: string): Promise<string> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await invokeOnce(modelId, query, intent, context);
    } catch (err) {
      lastErr = err;
      const name = (err as { name?: string }).name || "";
      const transient = TRANSIENT_ERR.test(name) || /throttl/i.test(String((err as Error).message));
      if (!transient || attempt === MAX_RETRIES) break;
      const delay = BASE_BACKOFF_MS * Math.pow(2, attempt) + Math.floor(Math.random() * 100);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw lastErr;
}

function parseTypedLines(output: string, query: string, includeLexical: boolean): Queryable[] {
  const lines = output.trim().split(/\r?\n/);
  const queryables: Queryable[] = [];
  const queryLower = query.toLowerCase();
  const queryTerms = queryLower
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
  const hasQueryTerm = (text: string): boolean => {
    if (queryTerms.length === 0) return true;
    const lower = text.toLowerCase();
    return queryTerms.some((t) => lower.includes(t));
  };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    const colon = line.indexOf(":");
    if (colon < 0) continue;
    const type = line.slice(0, colon).trim().toLowerCase();
    const content = line.slice(colon + 1).trim();
    if (!content) continue;
    if (type === "lex" || type === "vec" || type === "hyde") {
      if (!includeLexical && type === "lex") continue;
      queryables.push({ type: type as Queryable["type"], text: content });
    }
  }
  // Always include the original query as a vec variant if the model didn't.
  if (!queryables.some((q) => q.type === "vec" && hasQueryTerm(q.text))) {
    queryables.push({ type: "vec", text: query });
  }
  return queryables;
}

export async function bedrockExpandQuery(
  query: string,
  options: { context?: string; includeLexical?: boolean; intent?: string } = {}
): Promise<Queryable[]> {
  const includeLexical = options.includeLexical ?? true;
  const modelId = bedrockExpandModel();
  try {
    const text = await expandWithRetry(modelId, query, options.intent, options.context);
    const parsed = parseTypedLines(text, query, includeLexical);
    if (parsed.length === 0) {
      // Defensive: if the model returned nothing usable, fall back to the original query.
      return [{ type: "vec", text: query }];
    }
    return parsed;
  } catch (err) {
    console.error(`Bedrock expandQuery error (${modelId}):`, (err as Error).message);
    return [{ type: "vec", text: query }];
  }
}
