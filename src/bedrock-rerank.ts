/**
 * bedrock-rerank.ts - Optional AWS Bedrock reranking backend for QMD.
 *
 * Activated when QMD_RERANK_BACKEND=bedrock. Routes rank/rerank calls
 * through Cohere Rerank 3.5 on Bedrock instead of the local Qwen3-Reranker
 * GGUF model — eliminating the CPU bottleneck that causes MCP stdio
 * transport timeouts on `qmd query`.
 *
 * Default model: cohere.rerank-v3-5:0
 * Override with QMD_RERANK_MODEL.
 *
 * Region resolves from QMD_BEDROCK_REGION → AWS_REGION → us-east-1.
 */

import type { RerankResult, RerankDocument, RerankDocumentResult, RerankOptions } from "./llm.js";

const DEFAULT_BEDROCK_RERANK_MODEL = "cohere.rerank-v3-5:0";
const DEFAULT_BEDROCK_REGION = "us-east-1";
const MAX_RETRIES = 4;
const BASE_BACKOFF_MS = 250;
// Cohere Rerank caps documents per call. Be conservative.
const MAX_DOCS_PER_CALL = 1000;

export function isBedrockRerankEnabled(): boolean {
  return (process.env.QMD_RERANK_BACKEND || "").trim().toLowerCase() === "bedrock";
}

export function bedrockRerankModel(): string {
  return process.env.QMD_RERANK_MODEL?.trim() || DEFAULT_BEDROCK_RERANK_MODEL;
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
      `QMD_RERANK_BACKEND=bedrock requires @aws-sdk/client-bedrock-runtime. ` +
        `Install it: npm i @aws-sdk/client-bedrock-runtime\n` +
        `(original error: ${(err as Error).message})`
    );
  }
  cachedClient = new mod.BedrockRuntimeClient({ region: bedrockRegion() }) as unknown as BedrockClient;
  CommandCtor = mod.InvokeModelCommand as unknown as NonNullable<typeof CommandCtor>;
  return { client: cachedClient, Cmd: CommandCtor };
}

const TRANSIENT_ERR = /^(Throttling|ServiceUnavailable|TimeoutError|RequestTimeout|InternalServer)/i;

interface CohereRerankResponse {
  results?: Array<{ index: number; relevance_score: number }>;
}

async function invokeRerankOnce(
  modelId: string,
  query: string,
  documents: string[]
): Promise<Array<{ index: number; score: number }>> {
  const { client, Cmd } = await getClient();
  // Cohere Rerank 3.5 on Bedrock: { query, documents, top_n, api_version: 2 }
  // documents can be array of strings (preferred) or array of {text}.
  const body = {
    api_version: 2,
    query,
    documents,
    top_n: documents.length,
  };
  const cmd = new Cmd({
    modelId,
    contentType: "application/json",
    accept: "application/json",
    body: new TextEncoder().encode(JSON.stringify(body)),
  });
  const res = await client.send(cmd);
  const json = JSON.parse(new TextDecoder().decode(res.body)) as CohereRerankResponse;
  if (!json.results || !Array.isArray(json.results)) {
    throw new Error(`Bedrock Cohere Rerank response missing results (model=${modelId})`);
  }
  return json.results.map((r) => ({ index: r.index, score: r.relevance_score }));
}

async function rerankWithRetry(
  modelId: string,
  query: string,
  documents: string[]
): Promise<Array<{ index: number; score: number }>> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await invokeRerankOnce(modelId, query, documents);
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

export async function bedrockRerank(
  query: string,
  documents: RerankDocument[],
  _options: RerankOptions = {}
): Promise<RerankResult> {
  // Always use the Bedrock-configured model. Ignore caller's options.model —
  // it carries the upstream local-model id and would be rejected by Bedrock.
  const modelId = bedrockRerankModel();
  if (documents.length === 0) {
    return { results: [], model: modelId };
  }

  // Cohere has a per-call doc cap; chunk if needed and re-rank within each chunk.
  // For our typical workload (top 30 candidates) this is never hit.
  const chunks: RerankDocument[][] = [];
  for (let i = 0; i < documents.length; i += MAX_DOCS_PER_CALL) {
    chunks.push(documents.slice(i, i + MAX_DOCS_PER_CALL));
  }

  const allResults: RerankDocumentResult[] = [];
  for (let chunkOffset = 0, ci = 0; ci < chunks.length; chunkOffset += chunks[ci]!.length, ci++) {
    const chunk = chunks[ci]!;
    const texts = chunk.map((d) => d.text);
    let scored: Array<{ index: number; score: number }>;
    try {
      scored = await rerankWithRetry(modelId, query, texts);
    } catch (err) {
      console.error(`Bedrock rerank error (${modelId}):`, (err as Error).message);
      // On failure, return docs unchanged with neutral scores so the search still completes.
      for (let i = 0; i < chunk.length; i++) {
        allResults.push({ file: chunk[i]!.file, score: 0, index: chunkOffset + i });
      }
      continue;
    }
    for (const r of scored) {
      const orig = chunk[r.index];
      if (!orig) continue;
      allResults.push({ file: orig.file, score: r.score, index: chunkOffset + r.index });
    }
  }

  // Sort high-score-first (matches the contract callers expect).
  allResults.sort((a, b) => b.score - a.score);
  return { results: allResults, model: modelId };
}
