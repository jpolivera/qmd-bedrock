# qmd-bedrock

Cloudacio's fork of [`@tobilu/qmd`](https://github.com/tobi/qmd) that adds an
AWS Bedrock embedding backend, selectable at runtime via an environment
variable. All other qmd capabilities — file indexing, BM25 full-text search,
hybrid query, query expansion, and the Qwen3 reranker — remain on-device and
unchanged.

The default behavior matches upstream qmd. Bedrock kicks in only when you
opt in.

## Why this fork exists

Upstream qmd embeds documents using `node-llama-cpp` running
`embeddinggemma-300M` on CPU. On a typical EC2 host that's roughly
**1 chunk/sec**, which means a periodic reindex of even a few thousand chunks
saturates the box for several minutes. In our deployment that periodically
starved the OpenClaw gateway co-located on the same instance, causing Slack
sockets to flap and CommandCenter WebSocket handshakes to time out.

Routing only the embedder to Bedrock `titan-embed-text-v2` gets us roughly
**~13 ms/chunk** with 16-way concurrency, while keeping qmd's BM25, vector
storage, query expansion, and reranker entirely on-device.

| Workload | Local CPU (upstream) | Bedrock (this fork) |
|---|---|---|
| Per-chunk embed | ~10,000 ms | ~13 ms |
| 25,729-chunk full reindex | ~71 hours (extrapolated) | 5m 33s (measured) |

## Enabling the Bedrock backend

```bash
export QMD_EMBED_BACKEND=bedrock
export QMD_EMBED_MODEL=amazon.titan-embed-text-v2:0   # default; override if needed
export AWS_REGION=us-east-1                           # or QMD_BEDROCK_REGION
# AWS credentials resolve via the standard SDK chain (IMDS on EC2,
# AWS_PROFILE locally, env vars in containers). No qmd-specific config.

qmd embed
```

Optional knobs:

- `QMD_BEDROCK_DIMENSIONS` — `256`, `512`, or `1024`. Default: model default
  (1024 for titan-embed-text-v2).
- Both `amazon.titan-embed-*` and `cohere.embed-*` model IDs are recognized
  and given the right request body shape.

## How it works (one paragraph)

`src/llm.ts` checks `QMD_EMBED_BACKEND` at construction time. When set to
`bedrock`, `LlamaCpp.embed()`/`embedBatch()` short-circuit to a new
`bedrockEmbed`/`bedrockEmbedBatch` (in `src/bedrock-embed.ts`) that calls
Bedrock's `InvokeModelCommand`. The local llama model is never loaded for
embedding, eliminating the cold-start cost on every CLI invocation.
`tokenize()` and `countTokens()` fall back to a 4-chars-per-token heuristic
under the Bedrock backend, since the chunker only consults `.length`.
`detokenize()` returns empty (its fallback path is unreachable under
Bedrock's 50K-token input ceiling).

The AWS SDK is declared as an `optionalDependencies` entry, so users who
stay on the local backend pay no install cost.

## Migration notes

`amazon.titan-embed-text-v2:0` produces 1024-dim vectors, while
`embeddinggemma-300M` produces 768-dim. Vector dimensions are fixed per
sqlite-vec table, so switching backends requires a one-time wipe and rebuild
of `index.sqlite`.

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

## Required IAM permissions

The Bedrock InvokeModel permission must allow whatever embedding model you
configure. For the default:

```
bedrock:InvokeModel on arn:aws:bedrock:<region>::foundation-model/amazon.titan-embed-text-v2:0
```

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

Diff vs upstream is intentionally small:

- `+ src/bedrock-embed.ts`         (new — Bedrock client + retries)
- `~ src/llm.ts`                   (Bedrock short-circuit + tokenizer heuristic)
- `~ src/db.ts`                    (one-line Database type fix; upstream type gap, unrelated to Bedrock)
- `~ package.json`                 (name, description, repo URLs; +`@aws-sdk/client-bedrock-runtime` as optional dep)

License remains MIT, original copyright Tobi Lutke.
