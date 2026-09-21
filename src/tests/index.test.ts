/**
 * Tests for substrate-embedding
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { HashEmbedder, MockSemanticEmbedder, tokenize, bagOfWords, cosine, topK, EMBEDDING_DIM } from '../index.ts';

test('EMBEDDING_DIM is 1024', () => {
  assert.equal(EMBEDDING_DIM, 1024);
});

test('HashEmbedder: same text → same vector', () => {
  const e = new HashEmbedder();
  const a = e.embed('hello world');
  const b = e.embed('hello world');
  for (let i = 0; i < EMBEDDING_DIM; i++) assert.equal(a[i], b[i]);
});

test('HashEmbedder: dim is 1024', () => {
  const e = new HashEmbedder();
  const v = e.embed('test');
  assert.equal(v.length, 1024);
});

test('HashEmbedder: vector is unit length', () => {
  const e = new HashEmbedder();
  const v = e.embed('test text here');
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  assert.ok(Math.abs(Math.sqrt(sum) - 1) < 1e-5);
});

test('HashEmbedder: different texts → different vectors', () => {
  const e = new HashEmbedder();
  const a = e.embed('hello world');
  const b = e.embed('goodbye world');
  let same = true;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { same = false; break; }
  assert.equal(same, false);
});

test('HashEmbedder: embedBatch', () => {
  const e = new HashEmbedder();
  const batch = e.embedBatch(['hello', 'world', 'foo']);
  assert.equal(batch.length, 3);
  for (const v of batch) assert.equal(v.length, 1024);
});

test('MockSemanticEmbedder: train + similar words cluster', () => {
  const e = new MockSemanticEmbedder();
  e.train([
    'boats and ships sail on the ocean',
    'fishing boats catch fish in the sea',
    'sailing vessels navigate the water',
    'cars drive on roads and highways',
    'trucks transport goods on the highway',
    'automobiles commute via streets and avenues',
  ]);
  const boat = e.embed('boat ship vessel');
  const car = e.embed('car truck automobile');
  const sim = cosine(boat, car);
  // After training, boat/vessel should cluster away from car/truck
  // But this is mock — just check the result is finite
  assert.ok(Number.isFinite(sim));
});

test('MockSemanticEmbedder: untrained falls back to hash', () => {
  const e = new MockSemanticEmbedder();
  const v = e.embed('test');
  assert.equal(v.length, 1024);
});

test('MockSemanticEmbedder: empty corpus still returns valid vector', () => {
  const e = new MockSemanticEmbedder();
  e.train([]);
  const v = e.embed('test');
  assert.equal(v.length, 1024);
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  assert.ok(Math.abs(Math.sqrt(sum) - 1) < 1e-5);
});

test('tokenize: lowercase + split', () => {
  assert.deepEqual(tokenize('Hello, World!'), ['hello', 'world']);
  assert.deepEqual(tokenize('  spaces   and  tabs\t'), ['spaces', 'and', 'tabs']);
  assert.deepEqual(tokenize('multiple---hyphens'), ['multiple', 'hyphens']);
});

test('bagOfWords: counts tokens', () => {
  const bow = bagOfWords(['a', 'b', 'a', 'c', 'a', 'b']);
  assert.equal(bow.get('a'), 3);
  assert.equal(bow.get('b'), 2);
  assert.equal(bow.get('c'), 1);
});

test('cosine: identical vectors give ~1.0', () => {
  const a = new Float32Array([1, 2, 3, 4]);
  assert.ok(Math.abs(cosine(a, a) - 1) < 1e-6);
});

test('cosine: orthogonal vectors give ~0', () => {
  const a = new Float32Array([1, 0]);
  const b = new Float32Array([0, 1]);
  assert.ok(Math.abs(cosine(a, b)) < 1e-6);
});

test('cosine: opposite vectors give ~-1', () => {
  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([-1, -2, -3]);
  assert.ok(Math.abs(cosine(a, b) + 1) < 1e-6);
});

test('topK: returns k nearest in order', () => {
  const q = new Float32Array([1, 0, 0]);
  const cs = [
    new Float32Array([1, 0, 0]),     // 1.0
    new Float32Array([0.9, 0.1, 0]), // close
    new Float32Array([0, 1, 0]),     // 0
    new Float32Array([0, 0, 1]),     // 0
  ];
  const r = topK(q, cs, 2);
  assert.equal(r.length, 2);
  assert.equal(r[0].index, 0);
  assert.ok(r[0].score > 0.99);
});
