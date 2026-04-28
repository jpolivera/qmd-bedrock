# qmd-bedrock

Cloudacio's fork of [`@tobilu/qmd`](https://github.com/tobi/qmd) that adds
AWS Bedrock backends for the three runtime LLM operations qmd performs:
**embedding**, **query expansion**, and **reranking**. Each backend is
independently selectable at runtime via environment variables. All other qmd
capabilities — file indexing, BM25 full-text search, sqlite-vec storage,
hybrid query orchestration, MCP server — remain on-device and unchanged.

The default behavior matches upstream qmd. Bedrock kicks in only when you
opt in (per backend).

## Why this fork exists

Upstream qmd runs three local GGUF models via `node-llama-cpp`:

| Operation | Upstream local model | Approx CPU latency (4 vCPU, no GPU) |
|---|---|---|
| Embedding | `embeddinggemma-300M` | ~1,000 ms/chunk |
| Query expansion | `Qwen3-1.7B-q4_k_m` | ~20,000 ms |
| Reranking | `Qwen3-Reranker-0.6B` | ~20,000 ms (top-30) |

In our deployment, that combined cost made `qmd query` exceed mcporter's
stdio transport timeout, and a periodic reindex starved the co-located
OpenClaw gateway's event loop — Slack sockets flapped and CommandCenter
WebSocket handshakes timed out.

Routing all three operations to Bedrock keeps BM25 and sqlite-vec on-device
(near-zero latency, no cloud calls) but eliminates the CPU bottleneck:

| Operation | Local CPU | Bedrock | Default Bedrock model |
|---|---|---|---|
| Per-chunk embed | ~10,000 ms | ~13 ms | `amazon.titan-embed-text-v2:0` |
| 25,729-chunk full reindex | ~71 hours (extrapolated) | 5m 33s (measured) | (same as above) |
| Query expansion | ~20-21 s | ~720 ms | `amazon.nova-micro-v1:0` |
| Reranking 40 chunks | ~20 s | ~500 ms | `cohere.rerank-v3-5:0` |
| Total `qmd query` | timeouts | ~2.3 s | (composition of all three) |

## Enabling backends

Each backend is gated by its own pair of env vars (one for the switch, one
for the model id). Enable only what you need:

```bash
# Embedding (replaces embeddinggemma-300M)
export QMD_EMBED_BACKEND=bedrock
export QMD_EMBED_MODEL=amazon.titan-embed-text-v2:0   # default

# Reranking (replaces Qwen3-Reranker-0.6B)
export QMD_RERANK_BACKEND=bedrock
export QMD_RERANK_MODEL=cohere.rerank-v3-5:0          # default

# Query expansion (replaces Qwen3-1.7B)
export QMD_EXPAND_BACKEND=bedrock
export QMD_EXPAND_MODEL=amazon.nova-micro-v1:0        # default

# AWS region (required)
export AWS_REGION=us-east-1                           # or QMD_BEDROCK_REGION
```

When all three are set, qmd never loads the local GGUF models at all. You
can remove cached `.gguf` files from `~/.cache/qmd/models/` to reclaim disk
space (typically ~1.7 GB).

Credentials resolve via the standard AWS SDK chain (IMDS on EC2, `AWS_PROFILE`
locally, env vars in containers). No qmd-specific config required.

## Optional model knobs

**Embedding:**

- `QMD_BEDROCK_DIMENSIONS` — `256`, `512`, or `1024`. Default: model default
  (1024 for titan-embed-text-v2).
- Recognized model families: `amazon.titan-embed-*`, `cohere.embed-*`. The
  fork applies the right request body shape per family.

**Reranking:**

- Cohere Rerank 3.5 is currently the only supported family on Bedrock; the
  default is what's available.
- The fork ignores `options.model` from upstream callers (which carries an
  Ollama or HF model id from upstream config) and always uses
  `QMD_RERANK_MODEL`.

**Query expansion:**

- Recognized model families: `amazon.nova-*`, `anthropic.claude-*` (with a
  generic prompt-style fallback). The fork applies the right messages shape
  per family.
- The system prompt asks the model for `lex:`/`vec:`/`hyde:` typed lines so
  the parsed output matches the local expander's `Queryable[]` shape.

## How it works (under the hood)

`src/llm.ts` checks each `QMD_*_BACKEND` env var at `LlamaCpp` construction
time. When set to `bedrock`:

- `LlamaCpp.embed()` / `embedBatch()` short-circuit to `bedrockEmbed` /
  `bedrockEmbedBatch` (in `src/bedrock-embed.ts`).
- `LlamaCpp.rerank()` short-circuits to `bedrockRerank` (in
  `src/bedrock-rerank.ts`).
- `LlamaCpp.expandQuery()` short-circuits to `bedrockExpandQuery` (in
  `src/bedrock-expand.ts`).
- `tokenize()` / `countTokens()` fall back to a 4-chars-per-token heuristic
  under the embedding-Bedrock backend, since the chunker only consults
  `.length`. `detokenize()` returns empty (its fallback path is unreachable
  under Bedrock's input ceiling).

The local llama model is never loaded for any short-circuited operation,
eliminating cold-start cost on every CLI invocation.

The AWS SDK is declared as an `optionalDependencies` entry, so users who
stay entirely on local backends pay no install cost.

## `qmd status` reflects active backend

The status command now shows the active backend per operation:

```
Models
  Embedding:   AWS Bedrock — amazon.titan-embed-text-v2:0
  Reranking:   AWS Bedrock — cohere.rerank-v3-5:0
  Generation:  AWS Bedrock — amazon.nova-micro-v1:0
```

If you see `huggingface.co/...` (local CPU) instead of `AWS Bedrock`, the
relevant `QMD_*_BACKEND` env var didn't reach the qmd process — common
cause is forgetting to export it in the shell or systemd unit file.

## Migration: switching backends and the index

`amazon.titan-embed-text-v2:0` produces **1024-dim** vectors;
`embeddinggemma-300M` produces **768-dim**. Vector dimensions are fixed per
sqlite-vec table, so switching the **embedding** backend requires a one-time
wipe and rebuild of `index.sqlite`.

The cleanest path:

```bash
# stop services first, e.g. qmd.service
rm /home/openclaw/.cache/qmd/index.sqlite*
# start qmd, then:
qmd update    # rebuilds the index from your file collections
qmd embed     # generates fresh 1024-dim vectors via Bedrock
```

Document content is regenerable from your indexed file collections, so this
operation is non-destructive in practice.

**`qmd embed -f` is not sufficient** because sqlite-vec virtual tables
refuse to drop with `SQL logic error` even with the `vec0` extension loaded.
Removing `index.sqlite` and rebuilding is the working migration path.

The **rerank** and **expand** backends require no migration — they're
runtime-only and don't persist state.

## Required IAM permissions

If you stick with the default model ids:

```
bedrock:InvokeModel on:
  arn:aws:bedrock:<region>::foundation-model/amazon.titan-embed-text-v2:0
  arn:aws:bedrock:<region>::foundation-model/cohere.rerank-v3-5:0
  arn:aws:bedrock:<region>::foundation-model/amazon.nova-micro-v1:0
```

(Adjust ARNs if you override the default model IDs. AWS-managed
`BedrockFullAccess` covers all of the above.)

## Upstream relationship

Branch tracking:

- `main` — clean upstream qmd, kept in sync via `git pull upstream main`.
- `bedrock` — this fork's changes on top of `main`.

To rebase onto a newer upstream:

```bash
git fetch upstream
git rebase upstream/main bedrock
pnpm install && pnpm run build && pnpm test
```

Diff vs upstream is intentionally focused:

- `+ src/bedrock-embed.ts` — Bedrock embedding client + 16-way concurrency
- `+ src/bedrock-rerank.ts` — Cohere Rerank 3.5 client
- `+ src/bedrock-expand.ts` — Bedrock chat client for query expansion
- `~ src/llm.ts` — short-circuits in `embed`/`embedBatch`/`rerank`/`expandQuery`/`tokenize`/`countTokens`/`detokenize`
- `~ src/cli/qmd.ts` — `qmd status` reflects active backend per operation
- `~ src/db.ts` — one-line type fix on the custom `Database` interface (upstream type gap, unrelated to Bedrock)
- `~ package.json` — name, description, repo URLs; `@aws-sdk/client-bedrock-runtime` added as optional dep

License remains MIT, original copyright Tobi Lutke.
