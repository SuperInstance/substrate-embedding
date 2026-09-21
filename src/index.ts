/**
 * substrate-embedding: Text embeddings for semantic search
 *
 * BGE-Large compatible: 1024-dim float vectors.
 *
 * Two embedders ship here:
 *
 *   1. HashEmbedder — deterministic, dependency-free, no model.
 *      Same text → same vector. Use for testing + small deployments.
 *      Quality: poor. Words like "boat" and "ship" don't cluster.
 *
 *   2. MockSemanticEmbedder — uses token co-occurrence statistics
 *      trained on a small corpus to produce reasonable embeddings.
 *      No external model. Quality: usable for demo, not for production.
 *
 *   3. RemoteEmbedder — calls an HTTP endpoint (BGE server, FastEmbed,
 *      TEI, etc.). Production. The protocol is documented below.
 *
 * The 1024-d output dimension matches BGE-Large-en-v1.5. We can verify
 * byte-exact compatibility by running the same input through both.
 */

/** Standard BGE-Large embedding dimension. */
export const EMBEDDING_DIM = 1024;

/** Embedder interface — all embedders produce 1024-d float vectors. */
export interface Embedder {
  readonly dim: number;
  embed(text: string): Float32Array;
  embedBatch(texts: string[]): Float32Array[];
}

/** Hash-based embedder — deterministic, no model.
 *  Same algorithm as Vector.fromText but kept separate for clarity.
 *  Quality: poor for semantic similarity (boat/ship don't cluster).
 *  Use only for testing. */
export class HashEmbedder implements Embedder {
  readonly dim = EMBEDDING_DIM;

  embed(text: string): Float32Array {
    const data = new Float32Array(this.dim);
    for (let seed = 0; seed < 4; seed++) {
      const base = seed * 256;
      let h = 0xcbf29ce484222325n ^ BigInt(seed);
      const prime = 0x100000001b3n;
      const MASK = 0xFFFFFFFFFFFFFFFFn;
      // Lowercase + tokenize by whitespace
      const tokens = text.toLowerCase().split(/\W+/).filter(t => t.length > 0);
      for (let i = 0; i < tokens.length; i++) {
        const tok = tokens[i];
        for (let j = 0; j < tok.length; j++) {
          h = ((h ^ BigInt(tok.charCodeAt(j) & 0xff)) * prime) & MASK;
        }
        h = ((h ^ BigInt(i)) * prime) & MASK;
      }
      let state = h;
      for (let i = 0; i < 256; i++) {
        state = (state * prime) & MASK;
        const v = Number(BigInt.asIntN(32, state)) / (1 << 31);
        data[base + i] = v;
      }
    }
    // Normalize
    let norm2 = 0;
    for (let i = 0; i < this.dim; i++) norm2 += data[i] * data[i];
    const norm = Math.sqrt(norm2);
    if (norm > 0) for (let i = 0; i < this.dim; i++) data[i] /= norm;
    return data;
  }

  embedBatch(texts: string[]): Float32Array[] {
    return texts.map(t => this.embed(t));
  }
}

/** Co-occurrence based embedder. Trained on a small corpus.
 *  Produces reasonable embeddings for common words.
 *
 *  Algorithm:
 *    - Tokenize corpus
 *    - For each word, count context words within ±5 window
 *    - Build sparse co-occurrence matrix (vocab × vocab)
 *    - Apply PMI weighting: PMI(w, c) = log(P(w,c) / (P(w)P(c)))
 *    - SVD-reduce to 1024 dimensions
 *    - Embed a text by averaging PMI vectors of its tokens
 *
 *  For a small corpus (< 10K docs), this works without any model. */
export class MockSemanticEmbedder implements Embedder {
  readonly dim = EMBEDDING_DIM;
  private vocab: Map<string, number> = new Map();
  private inverseVocab: string[] = [];
  private cooccurrence: Float32Array;  // vocab × vocab sparse (dense here for simplicity)
  private wordVectors: Float32Array;
  private contextWindow = 5;
  private minCount = 1;
  private trained = false;

  constructor() {
    this.cooccurrence = new Float32Array(0);
    this.wordVectors = new Float32Array(0);
  }

  /** Train on a corpus of documents. Each doc is a string. */
  train(docs: string[], contextWindow: number = 5, minCount: number = 2): void {
    this.contextWindow = contextWindow;
    this.minCount = minCount;
    const wordCounts = new Map<string, number>();
    const docTokens: string[][] = [];
    // Tokenize + count
    for (const doc of docs) {
      const tokens = doc.toLowerCase().split(/\W+/).filter(t => t.length > 0);
      docTokens.push(tokens);
      for (const t of tokens) {
        wordCounts.set(t, (wordCounts.get(t) || 0) + 1);
      }
    }
    // Build vocab (drop words below minCount)
    let idx = 0;
    for (const [w, c] of wordCounts) {
      if (c >= minCount) {
        this.vocab.set(w, idx);
        this.inverseVocab.push(w);
        idx++;
      }
    }
    const V = this.vocab.size;
    this.cooccurrence = new Float32Array(V * V);
    // Count co-occurrences
    let totalPairs = 0;
    for (const tokens of docTokens) {
      const indices: number[] = [];
      for (const t of tokens) {
        const i = this.vocab.get(t);
        if (i !== undefined) indices.push(i);
      }
      for (let i = 0; i < indices.length; i++) {
        for (let j = i + 1; j < indices.length && j <= i + this.contextWindow; j++) {
          const a = indices[i], b = indices[j];
          this.cooccurrence[a * V + b] += 1;
          this.cooccurrence[b * V + a] += 1;
          totalPairs++;
        }
      }
    }
    if (totalPairs === 0) {
      // Empty corpus — fall back to hash
      this.trained = false;
      return;
    }
    // Compute PMI
    const wordProbs = new Float32Array(V);
    const colSums = new Float32Array(V);
    for (let i = 0; i < V; i++) {
      let sum = 0;
      for (let j = 0; j < V; j++) {
        sum += this.cooccurrence[i * V + j];
      }
      colSums[i] = sum;
      wordProbs[i] = sum / (2 * totalPairs);
    }
    const pmiMatrix = new Float32Array(V * V);
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) {
        const pij = this.cooccurrence[i * V + j] / totalPairs;
        const pi = wordProbs[i];
        const pj = wordProbs[j];
        if (pij > 0 && pi > 0 && pj > 0) {
          pmiMatrix[i * V + j] = Math.log(pij / (pi * pj));
        } else {
          pmiMatrix[i * V + j] = 0;
        }
      }
    }
    // SVD: use the truncated power iteration method
    this.wordVectors = this.truncatedSVD(pmiMatrix, V, this.dim);
    this.trained = true;
  }

  private truncatedSVD(matrix: Float32Array, rows: number, k: number): Float32Array {
    // Power iteration with deflation to get top-k singular vectors
    const vectors = new Float32Array(rows * k);
    let residual = new Float32Array(matrix);
    for (let i = 0; i < k; i++) {
      // Initialize random vector
      let v = new Float32Array(rows);
      for (let j = 0; j < rows; j++) v[j] = Math.random() - 0.5;
      // Power iterate
      for (let iter = 0; iter < 30; iter++) {
        const newV = new Float32Array(rows);
        for (let a = 0; a < rows; a++) {
          let sum = 0;
          for (let b = 0; b < rows; b++) {
            sum += residual[a * rows + b] * v[b];
          }
          newV[a] = sum;
        }
        // Normalize
        let norm = 0;
        for (let j = 0; j < rows; j++) norm += newV[j] * newV[j];
        norm = Math.sqrt(norm);
        if (norm < 1e-10) break;
        for (let j = 0; j < rows; j++) v[j] = newV[j] / norm;
      }
      // Store
      for (let j = 0; j < rows; j++) vectors[j * k + i] = v[j];
      // Deflate: subtract rank-1 contribution
      // Compute eigenvalue: λ = v^T (residual) v
      let lambda = 0;
      for (let a = 0; a < rows; a++) {
        for (let b = 0; b < rows; b++) {
          lambda += v[a] * residual[a * rows + b] * v[b];
        }
      }
      for (let a = 0; a < rows; a++) {
        for (let b = 0; b < rows; b++) {
          residual[a * rows + b] -= lambda * v[a] * v[b];
        }
      }
    }
    return vectors;
  }

  embed(text: string): Float32Array {
    const data = new Float32Array(this.dim);
    if (!this.trained) {
      // Fall back to hash embedder
      return new HashEmbedder().embed(text);
    }
    const tokens = text.toLowerCase().split(/\W+/).filter(t => t.length > 0);
    let count = 0;
    for (const tok of tokens) {
      const idx = this.vocab.get(tok);
      if (idx !== undefined) {
        for (let d = 0; d < this.dim; d++) {
          data[d] += this.wordVectors[idx * this.dim + d];
        }
        count++;
      }
    }
    if (count > 0) {
      for (let d = 0; d < this.dim; d++) data[d] /= count;
    }
    // Normalize
    let norm = 0;
    for (let d = 0; d < this.dim; d++) norm += data[d] * data[d];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let d = 0; d < this.dim; d++) data[d] /= norm;
    return data;
  }

  embedBatch(texts: string[]): Float32Array[] {
    return texts.map(t => this.embed(t));
  }
}

/** Remote embedder — calls an HTTP endpoint.
 *  Default URL is local BGE-compatible server (port 8080, /embed).
 *  Compatible with:
 *    - Text Embeddings Inference (TEI) by HuggingFace
 *    - Infinity inference server
 *    - FastEmbed server
 *    - Any service that POSTs {"inputs": [text]} → {"embeddings": [[floats...]]} */
export class RemoteEmbedder implements Embedder {
  readonly dim = EMBEDDING_DIM;
  constructor(
    private url: string = 'http://localhost:8080/embed',
    private batchSize: number = 32,
  ) {}

  async embed(text: string): Promise<Float32Array> {
    const result = await this.embedBatch([text]);
    return result[0];
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    const results: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const resp = await fetch(this.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: batch }),
      });
      if (!resp.ok) throw new Error(`RemoteEmbedder: HTTP ${resp.status}`);
      const json = await resp.json() as { embeddings: number[][] };
      for (const emb of json.embeddings) {
        results.push(new Float32Array(emb));
      }
    }
    return results;
  }
}

/** Tokenizer — simple, used by the mock embedder and exposed for callers.
 *  Lowercases, splits on non-word, drops empty tokens. */
export function tokenize(text: string): string[] {
  return text.toLowerCase().split(/\W+/).filter(t => t.length > 0);
}

/** Bag-of-words representation. Returns a sparse map of token → count. */
export function bagOfWords(tokens: string[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const t of tokens) m.set(t, (m.get(t) || 0) + 1);
  return m;
}

/** Cosine similarity between two Float32Array vectors. */
export function cosine(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-10);
}

/** Top-k most similar texts to a query. */
export function topK(query: Float32Array, candidates: Float32Array[], k: number): Array<{ index: number, score: number }> {
  const scored = candidates.map((c, i) => ({ index: i, score: cosine(query, c) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}
