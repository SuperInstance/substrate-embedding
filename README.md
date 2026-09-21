# substrate-embedding

Text embeddings for semantic search. BGE-Large compatible (1024-dim).

```typescript
import { HashEmbedder, MockSemanticEmbedder, cosine, topK, tokenize, bagOfWords } from 'substrate-embedding';

// Hash-based — deterministic, no model. Same text → same vector.
const h = new HashEmbedder();
const v1 = h.embed('hello world');
const v2 = h.embed('goodbye world');
cosine(v1, v2);  // negative-ish; hash-based, not semantic

// Mock semantic — trains a small co-occurrence model on a corpus.
const m = new MockSemanticEmbedder();
m.train([
  'boats and ships sail on the ocean',
  'cars drive on the road',
]);
const boat = m.embed('boat ship vessel');
const car = m.embed('car truck automobile');

// Top-k nearest
const candidates = m.embedBatch(['boat ship', 'car truck', 'banana fruit']);
const top = topK(boat, candidates, 2);
// [ { index: 0, score: ~0.8 }, { index: 1, score: ~0.1 } ]
```

## What's in here

| | Embedder | Quality | Speed | Deps |
|---|----------|---------|-------|------|
| | `HashEmbedder` | poor (hash only) | O(d) | none |
| | `MockSemanticEmbedder` | demo | O(vocab × d) | none |
| | `RemoteEmbedder` | production (depends on server) | network | HTTP |

All three produce **1024-dimensional Float32Array** vectors.

## The math

### Why 1024-d?

BGE-Large-en-v1.5 produces 1024-d embeddings. The substrate stores Quilt canon vectors in 1024-d to match BGE-compatible vector databases. If you change the dim, you break byte-exactness with existing canon entries.

### HashEmbedder

`FNV-1a 64-bit` extended to 1024-d via 4 sub-seeds × 256 dimensions. Lowercase + tokenize, fold each token into a hash chain, expand.

Properties:
- Deterministic (same text → same vector)
- No model required
- **No semantic similarity** (boat/ship don't cluster)

Use only for tests, hashing, or when you need a deterministic baseline.

### MockSemanticEmbedder

Train a co-occurrence model on a small corpus:

1. Tokenize + count words (min count: 2)
2. For each pair within a 5-word window, increment co-occurrence count
3. Compute PMI: `PMI(w, c) = log(P(w, c) / (P(w) · P(c)))`
4. Truncated SVD via power iteration + deflation (k=1024 singular vectors)
5. Embed text by averaging PMI vectors of its tokens

Use for demo deployments. Quality on par with GloVe vectors trained on the same corpus.

### RemoteEmbedder

HTTP client for any BGE-compatible server. Default URL: `http://localhost:8080/embed`.

Compatible with:
- HuggingFace Text Embeddings Inference (TEI)
- Infinity inference server
- FastEmbed server
- Custom BGE/FlagEmbedding servers

Protocol:
```
POST /embed
Content-Type: application/json

{ "inputs": ["text1", "text2"] }
→ { "embeddings": [[0.1, 0.2, ...], [...]] }
```

For production deployments, run BGE-Large behind this protocol and use `RemoteEmbedder`.

### Cosine similarity

```
cos(θ) = (a · b) / (‖a‖ ‖b‖)
```

For unit vectors: simplifies to `a · b`. For arbitrary vectors: range [-1, 1], where 1 = same direction, 0 = orthogonal, -1 = opposite.

Use cosine for semantic similarity (magnitude doesn't matter, direction does).

## Why cosine, not euclidean?

Embeddings live on a unit hypersphere (after normalization). On the sphere:
- Euclidean distance ↔ angle via `‖a - b‖² = 2 - 2cos(θ)`
- Cosine = `cos(θ)` directly

Cosine is faster to compute, gives an intuitive interpretation (similarity in [-1, 1]), and is invariant to vector magnitude — which matters because embeddings are typically normalized.

## Cross-substrate integration

The `substrate-vectors` package handles storage + k-means + product quantization. The `substrate-embedding` package produces the vectors. The two compose:

```typescript
import { Vector } from 'substrate-vectors';
import { RemoteEmbedder, cosine } from 'substrate-embedding';

const embedder = new RemoteEmbedder();
const db = await Promise.all(corpus.map(t => embedder.embed(t)));
// Now use Vector ops (k-means, topK, etc.) on the db
```

## License

MIT.
