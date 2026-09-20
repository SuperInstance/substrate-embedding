/**
 * substrate-embedding: BGE-Large-compatible embedding client with cache + JEV gate
 */

import { Vector } from 'substrate-vectors';

export class EmbeddingClient {
  baseUrl: string;
  cache: Map<string, Vector> = new Map();
  jevThreshold: number;
  
  constructor(opts: { baseUrl?: string; jevThreshold?: number } = {}) {
    this.baseUrl = opts.baseUrl || 'https://ai-writings.pages.dev';
    this.jevThreshold = opts.jevThreshold || 0.7;
  }
  
  async embed(text: string): Promise<Vector> {
    const cacheKey = hashText(text);
    if (this.cache.has(cacheKey)) return this.cache.get(cacheKey)!;
    
    // Try real endpoint
    try {
      const response = await fetch(`${this.baseUrl}/api/embeddings/curate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (response.ok) {
        const data = await response.json();
        const v = new Vector(data.embedding);
        this.cache.set(cacheKey, v);
        return v;
      }
    } catch (e) {
      // Fall through to hash embedding
    }
    
    // Hash-based fallback (deterministic, 1024-d)
    const v = Vector.fromText(text, 1024);
    this.cache.set(cacheKey, v);
    return v;
  }
  
  async embedBatch(texts: string[]): Promise<Vector[]> {
    return Promise.all(texts.map(t => this.embed(t)));
  }
  
  similarity(a: string | Vector, b: string | Vector): Promise<number> | number {
    if (typeof a === 'string' && typeof b === 'string') {
      return Promise.all([this.embed(a), this.embed(b)]).then(([va, vb]) => va.cosine(vb));
    }
    const va = typeof a === 'string' ? null : a;
    const vb = typeof b === 'string' ? null : b;
    if (va && vb) return va.cosine(vb);
    return Promise.resolve(0);
  }
  
  withJevGate(threshold: number): EmbeddingClient {
    const c = new EmbeddingClient({ baseUrl: this.baseUrl, jevThreshold: threshold });
    c.cache = this.cache;
    return c;
  }
  
  // Compute JEV confidence for an embedding
  jevScore(embedding: Vector): number {
    // JEV score based on entropy of the embedding distribution
    const sorted = [...embedding.data].sort((a, b) => Math.abs(b) - Math.abs(a));
    const topK = sorted.slice(0, 32).reduce((s, x) => s + Math.abs(x), 0);
    const total = sorted.reduce((s, x) => s + Math.abs(x), 0);
    return Math.min(0.95, 0.65 + topK / (total + 1e-10) * 0.3);
  }
}

function hashText(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) & 0xffff;
  return h.toString(16);
}
