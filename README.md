# qmd-bedrock

A fork of [`@tobilu/qmd`](https://github.com/tobi/qmd) that adds **AWS Bedrock
backends for the three LLM operations qmd uses**: embedding, query expansion,
and reranking. Each backend is independently selectable at runtime via env
vars. Everything else — file indexing, BM25 full-text search, hybrid query,
sqlite-vec storage — is unchanged and still on-device.

The default behavior matches upstream qmd. Bedrock kicks in only when you opt
in (per backend).

## Why

Upstream qmd runs three local GGUF models via `node-llama-cpp`:

- **embeddinggemma-300M** for vector embeddings
- **Qwen3-1.7B** for query expansion
- **Qwen3-Reranker-0.6B** for cross-encoder reranking

On a 4-vCPU host with no GPU, each is slow:

- Embedding: ~1 chunk/sec — a periodic reindex of a few thousand chunks
  saturates the box for several minutes
- Query expansion: ~20 seconds per `qmd query` call
- Reranking: another ~20 seconds for top-30 candidates

Combined, `qmd query` takes 40+ seconds on CPU — well past mcporter's stdio
transport timeout, making the MCP path practically unusable. And while these
loops run, any co-located service (in our case the OpenClaw gateway) gets
event-loop-starved.

Routing all three to Bedrock keeps qmd's BM25 + sqlite-vec storage local
(near-zero latency, no cloud calls) but eliminates the CPU bottleneck:

| Operation | Local CPU (upstream) | Bedrock (this fork) |
| --- | --- | --- |
| Per-chunk embed | ~10,000 ms | ~13 ms |
| 25,729-chunk full reindex | ~71 hours (extrapolated) | 5m 33s (measured) |
| Query expansion | ~20-21 s | ~720 ms |
| Reranking 40 chunks | ~20 s | ~500 ms |
| Total `qmd query` | timeouts | ~2.3 s |

## Enable it

Each backend is independent. Enable only what you need:

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

# AWS region
export AWS_REGION=us-east-1                           # or QMD_BEDROCK_REGION

qmd query "your search"
```

When all three are set, qmd never loads the local GGUF models at all. You
can remove the cached `.gguf` files from `~/.cache/qmd/models/` to reclaim
~1.7 GB of disk space.

Credentials resolve via the standard AWS SDK chain (IMDS on EC2,
`AWS_PROFILE` locally, env vars in containers). No qmd-specific config
required.

Optional embedding knobs:

- `QMD_BEDROCK_DIMENSIONS` — `256`, `512`, or `1024`. Default: model default
  (1024 for `titan-embed-text-v2`).
- Both `amazon.titan-embed-*` and `cohere.embed-*` model IDs are recognized
  for embedding, with the right request body shape applied per family.
- For query expansion, both `amazon.nova-*` and `anthropic.claude-*` model
  families are supported, plus a generic prompt-style fallback.

## Verify it's working

`qmd status` shows the active backend per operation:

```
Models
  Embedding:   AWS Bedrock — amazon.titan-embed-text-v2:0
  Reranking:   AWS Bedrock — cohere.rerank-v3-5:0
  Generation:  AWS Bedrock — amazon.nova-micro-v1:0
```

If you see `huggingface.co/...` (local CPU) instead of `AWS Bedrock`, the
relevant `QMD_*_BACKEND` env var isn't reaching the qmd process.

## Migration

`amazon.titan-embed-text-v2:0` produces 1024-dim vectors; `embeddinggemma-300M`
produces 768. Vector dimensions are fixed per `sqlite-vec` table, so switching
the embedding backend requires a one-time wipe and rebuild of `index.sqlite`.
See [docs/BEDROCK.md](docs/BEDROCK.md) for the full step-by-step.

Switching the rerank or expand backends requires no migration — they're
runtime-only operations and don't persist anything.

## What's different from upstream

| File | Change |
| --- | --- |
| `src/bedrock-embed.ts` | New — Bedrock embedding client (titan + cohere request shapes), retries, 16-way concurrency |
| `src/bedrock-rerank.ts` | New — Cohere Rerank 3.5 client, transient-error retries, doc chunking |
| `src/bedrock-expand.ts` | New — Bedrock chat client for query expansion (Nova + Anthropic body shapes) |
| `src/llm.ts` | Short-circuits in `embed`, `embedBatch`, `rerank`, `expandQuery`, `tokenize`, `countTokens`, `detokenize` when the matching `QMD_*_BACKEND=bedrock` is set |
| `src/cli/qmd.ts` | `qmd status` now reports the active backend per operation instead of always printing the local GGUF URL |
| `src/db.ts` | One-line type fix on the custom `Database` interface — upstream type gap, unrelated to Bedrock |
| `package.json` | Renamed, repo URLs updated, `@aws-sdk/client-bedrock-runtime` added as `optionalDependencies` |
| `docs/BEDROCK.md` | This fork's docs |

The AWS SDK lives under `optionalDependencies`, so users who stay on the
local backend pay no install cost.

## Branches

- `main` — clean upstream qmd, kept in sync via `git pull upstream main`.
- `bedrock` — this fork's changes on top of `main`. **Default branch.**

To rebase onto a newer upstream qmd:

```bash
git fetch upstream
git rebase upstream/main bedrock
pnpm install && pnpm run build && pnpm test
```

## Required IAM permissions

```
bedrock:InvokeModel on:
  arn:aws:bedrock:<region>::foundation-model/amazon.titan-embed-text-v2:0
  arn:aws:bedrock:<region>::foundation-model/cohere.rerank-v3-5:0
  arn:aws:bedrock:<region>::foundation-model/amazon.nova-micro-v1:0
```

(Adjust ARNs if you override the default model IDs.)

## Upstream README

The original qmd README is preserved at
[`README-UPSTREAM.md`](README-UPSTREAM.md) for full feature documentation,
search syntax, MCP integration, etc. — none of which has changed.

## License

MIT, original copyright Tobi Lutke.
