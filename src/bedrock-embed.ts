/**
 * bedrock-embed.ts - Optional AWS Bedrock embedding backend for QMD.
 *
 * Activated when QMD_EMBED_BACKEND=bedrock. Routes embed/embedBatch calls
 * through Amazon Bedrock InvokeModel instead of node-llama-cpp, avoiding
 * the cold model load and CPU inference cost on every CLI invocation.
 *
 * Model defaults to `amazon.titan-embed-text-v2:0` (1024-dim).
 * Override with QMD_EMBED_MODEL when backend=bedrock.
 *
 * Region resolves from QMD_BEDROCK_REGION → AWS_REGION → us-east-1.
 *
 * Auth uses the AWS SDK default credential chain (IMDS on EC2, profile
 * locally, env vars in containers). No qmd-specific config.
 */

import type { EmbeddingResult } from "./llm.js";

const DEFAULT_BEDROCK_MODEL = "amazon.titan-embed-text-v2:0";
const DEFAULT_BEDROCK_REGION = "us-east-1";
const MAX_CONCURRENCY = 16;
const MAX_RETRIES = 4;
const BASE_BACKOFF_MS = 250;

export function isBedrockEmbedEnabled(): boolean {
  return (process.env.QMD_EMBED_BACKEND || "").trim().toLowerCase() === "bedrock";
}

export function bedrockEmbedModel(): string {
  return process.env.QMD_EMBED_MODEL?.trim() || DEFAULT_BEDROCK_MODEL;
}

function bedrockRegion(): string {
  return (process.env.QMD_BEDROCK_REGION || process.env.AWS_REGION || DEFAULT_BEDROCK_REGION).trim();
}

// Lazy import: only require @aws-sdk/client-bedrock-runtime when actually used,
// so the local-only install path doesn't pull AWS SDK transitively.
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
      `QMD_EMBED_BACKEND=bedrock requires @aws-sdk/client-bedrock-runtime. ` +
        `Install it: npm i @aws-sdk/client-bedrock-runtime\n` +
        `(original error: ${(err as Error).message})`
    );
  }
  cachedClient = new mod.BedrockRuntimeClient({ region: bedrockRegion() }) as unknown as BedrockClient;
  CommandCtor = mod.InvokeModelCommand as unknown as NonNullable<typeof CommandCtor>;
  return { client: cachedClient, Cmd: CommandCtor };
}

const TRANSIENT_ERR = /^(Throttling|ServiceUnavailable|TimeoutError|RequestTimeout|InternalServer)/i;

async function invokeOnce(modelId: string, body: Record<string, unknown>): Promise<number[]> {
  const { client, Cmd } = await getClient();
  const cmd = new Cmd({
    modelId,
    contentType: "application/json",
    accept: "application/json",
    body: new TextEncoder().encode(JSON.stringify(body)),
  });
  const res = await client.send(cmd);
  const json = JSON.parse(new TextDecoder().decode(res.body)) as { embedding?: number[]; embeddings?: number[][] };
  // Titan v2 returns { embedding: number[] }; cohere.embed returns { embeddings: number[][] }
  const vec = json.embedding ?? (Array.isArray(json.embeddings) ? json.embeddings[0] : undefined);
  if (!Array.isArray(vec) || vec.length === 0) {
    throw new Error(`Bedrock response missing embedding (model=${modelId})`);
  }
  return vec;
}

function buildBody(modelId: string, text: string): Record<string, unknown> {
  // Titan models: { inputText, dimensions, normalize }
  // Cohere: { texts: [string], input_type: "search_document" }
  if (/^cohere\./i.test(modelId)) {
    return { texts: [text], input_type: "search_document" };
  }
  // titan-embed-text-v2 also accepts dimensions: 256|512|1024 and normalize: bool
  const body: Record<string, unknown> = { inputText: text };
  const dims = process.env.QMD_BEDROCK_DIMENSIONS;
  if (dims) {
    const parsed = Number.parseInt(dims, 10);
    if (Number.isInteger(parsed) && [256, 512, 1024].includes(parsed)) {
      body.dimensions = parsed;
    }
  }
  return body;
}

async function embedWithRetry(modelId: string, text: string): Promise<number[]> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await invokeOnce(modelId, buildBody(modelId, text));
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

export async function bedrockEmbed(text: string): Promise<EmbeddingResult | null> {
  const modelId = bedrockEmbedModel();
  try {
    const embedding = await embedWithRetry(modelId, text);
    return { embedding, model: modelId };
  } catch (err) {
    console.error(`Bedrock embed error (${modelId}):`, (err as Error).message);
    return null;
  }
}

export async function bedrockEmbedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
  if (texts.length === 0) return [];
  const modelId = bedrockEmbedModel();
  const out: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);
  let cursor = 0;
  const workers = Array.from({ length: Math.min(MAX_CONCURRENCY, texts.length) }, async () => {
    while (true) {
      const i = cursor++;
      if (i >= texts.length) return;
      try {
        const embedding = await embedWithRetry(modelId, texts[i]!);
        out[i] = { embedding, model: modelId };
      } catch (err) {
        console.error(`Bedrock embed error at index ${i} (${modelId}):`, (err as Error).message);
        out[i] = null;
      }
    }
  });
  await Promise.all(workers);
  return out;
}
