# qmd-bedrock

A fork of [`@tobilu/qmd`](https://github.com/tobi/qmd) that adds an **AWS
Bedrock embedding backend**, selectable at runtime. Everything else qmd does —
file indexing, BM25 full-text search, hybrid query, query expansion, and the
Qwen3 reranker — is unchanged and still runs on-device.

The default behavior matches upstream qmd. Bedrock kicks in only when you opt
in.

## Why

Upstream qmd embeds documents using `node-llama-cpp` running
`embeddinggemma-300M` on CPU. On a typical 4-vCPU host that's roughly
**1 chunk/sec**, which means a periodic reindex of even a few thousand chunks
saturates the box for several minutes. In our deployment, that periodically
starved the OpenClaw gateway co-located on the same instance, causing Slack
sockets to flap and CommandCenter WebSocket handshakes to time out.

Routing only the embedder to Bedrock `titan-embed-text-v2` gets us roughly
**~13 ms/chunk** with 16-way concurrency, while keeping qmd's other on-device
capabilities intact.

| Workload | Local CPU (upstream) | Bedrock (this fork) |
| --- | --- | --- |
| Per-chunk embed | ~10,000 ms | ~13 ms |
| 25,729-chunk full reindex | ~71 hours (extrapolated) | 5m 33s (measured) |

## Enable it

```bash
export QMD_EMBED_BACKEND=bedrock
export QMD_EMBED_MODEL=amazon.titan-embed-text-v2:0   # default; override if needed
export AWS_REGION=us-east-1                           # or QMD_BEDROCK_REGION

qmd embed
```

Credentials resolve via the standard AWS SDK chain (IMDS on EC2,
`AWS_PROFILE` locally, env vars in containers). No qmd-specific config
required.

Optional knobs:

- `QMD_BEDROCK_DIMENSIONS` — `256`, `512`, or `1024`. Default: model default
  (1024 for `titan-embed-text-v2`).
- Both `amazon.titan-embed-*` and `cohere.embed-*` model IDs are recognized
  and given the right request body shape.

## Migration

`amazon.titan-embed-text-v2:0` produces 1024-dim vectors; `embeddinggemma-300M`
produces 768. Vector dimensions are fixed per `sqlite-vec` table, so switching
backends requires a one-time wipe and rebuild of `index.sqlite`. See
[docs/BEDROCK.md](docs/BEDROCK.md) for the full step-by-step.

## What's different from upstream

Five-file diff:

| File | Change |
| --- | --- |
| `src/bedrock-embed.ts` | New — Bedrock client, request shaping, retries with exponential backoff, concurrency cap of 16 |
| `src/llm.ts` | Short-circuits `embed`/`embedBatch`/`tokenize`/`countTokens`/`detokenize` to Bedrock when `QMD_EMBED_BACKEND=bedrock` |
| `src/db.ts` | One-line type fix on the custom `Database` interface — upstream gap, unrelated to Bedrock |
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
bedrock:InvokeModel on arn:aws:bedrock:<region>::foundation-model/amazon.titan-embed-text-v2:0
```

## Upstream README

The original qmd README is preserved at
[`README-UPSTREAM.md`](README-UPSTREAM.md) for full feature documentation,
search syntax, MCP integration, etc. — none of which has changed.

## License

MIT, original copyright Tobi Lutke.
