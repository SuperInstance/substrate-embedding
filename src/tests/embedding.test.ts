import { test } from 'node:test';
import assert from 'node:assert';
import { HashEmbedder, MockSemanticEmbedder, EMBEDDING_DIM, cosine } from '../index.ts';

test('HashEmbedder returns 1024-dim vector', () => {
  const e = new HashEmbedder();
  const v = e.embed('hello world');
  assert.strictEqual(v.length, EMBEDDING_DIM);
});

test('HashEmbedder is deterministic', () => {
  const e = new HashEmbedder();
  const a = e.embed('hello');
  const b = e.embed('hello');
  const c = cosine(a, b);
  assert.ok(Math.abs(c - 1.0) < 1e-9, `expected ~1.0, got ${c}`);
});

test('HashEmbedder different texts differ', () => {
  const e = new HashEmbedder();
  const a = e.embed('hello');
  const b = e.embed('world');
  const c = cosine(a, b);
  assert.ok(c < 0.999, `expected <0.999, got ${c}`);
});

test('MockSemanticEmbedder returns 1024-dim vector', async () => {
  const e = new MockSemanticEmbedder();
  const v = await e.embed('hello world');
  assert.strictEqual(v.length, EMBEDDING_DIM);
});
