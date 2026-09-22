/**
 * Receipted training tests: the SVD's stochastic heart, hash-chained.
 *
 * MockSemanticEmbedder.train() with {ledger, uniform} books every SVD
 * component init as an EFFECT row (vendored 4quilt family recipe from
 * SuperInstance/substrate-rng). Same corpus + same seed must reproduce
 * the same embedding table — and the chain must say so.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { MockSemanticEmbedder, SeededUniform } from "../index.ts";
import { verifyChain } from "../ledger.ts";

const CORPUS = [
  "boats and ships sail on the ocean",
  "cars drive on the road",
  "sailboats and yachts are boats",
  "trucks and cars drive on highways",
  "the ocean is full of ships and boats",
];

function trainReceipted(seed: string) {
  const embedder = new MockSemanticEmbedder();
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { DrawLedger } = await_import();
  const ledger = new DrawLedger("substrate-embedding", `train-${seed}`);
  const uniform = new SeededUniform(seed);
  embedder.train(CORPUS, 5, 1, { receipt: { ledger, uniform } });
  return { embedder, ledger };
}

// Tiny indirection so the import sits at top of one place.
import { DrawLedger as _DrawLedger } from "../ledger.ts";
function await_import() {
  return { DrawLedger: _DrawLedger };
}

function tableHash(e: MockSemanticEmbedder): string {
  // Reach the trained table through a probe embedding is not enough;
  // hash the internal wordVectors via reflection on the private field.
  const wv = (e as unknown as { wordVectors: Float32Array }).wordVectors;
  const bytes = new Uint8Array(wv.buffer, wv.byteOffset, wv.byteLength);
  let h = 0x811c9dc5;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

test("seeded receipted train: chain verifies, shape is BIND…VIEW", () => {
  const { ledger } = trainReceipted("alpha");
  const [ok, bad, why] = ledger.verify();
  assert.equal(ok, true, `bad=${bad} why=${why}`);
  assert.equal(ledger.rows[0].op, "BIND");
  assert.equal(ledger.rows[ledger.rows.length - 1].op, "VIEW");
  const effects = ledger.rows.filter((r) => r.op === "EFFECT");
  const heartbeats = ledger.rows.filter((r) => r.op === "TICK");
  assert.equal(effects.length, 1024); // one per SVD component
  assert.equal(heartbeats.length, 8); // every 128 components
  assert.equal(
    ledger.rows.length,
    1 + effects.length + heartbeats.length + 1,
  );
  assert.equal(ledger.rows[0].payload.kind, "rng-run/v1");
  assert.equal(ledger.rows[1].payload.kind, "draw/v1");
  assert.equal(ledger.rows[1].payload.distribution, "svd_init_uniform");
});

test("same seed + same corpus → byte-identical trained table", () => {
  const a = trainReceipted("repeatable");
  const b = trainReceipted("repeatable");
  assert.equal(tableHash(a.embedder), tableHash(b.embedder));
  const va = a.embedder.embed("boats and ships");
  const vb = b.embedder.embed("boats and ships");
  for (let i = 0; i < va.length; i++) assert.equal(va[i], vb[i]);
});

test("different seed → different table (and chains diverge at row 1)", () => {
  const a = trainReceipted("seed-one");
  const b = trainReceipted("seed-two");
  assert.notEqual(tableHash(a.embedder), tableHash(b.embedder));
  assert.notEqual(a.ledger.rows[1].row_hash, b.ledger.rows[1].row_hash);
});

test("tampered draw row is caught with ROW_HASH_MISMATCH", () => {
  const { ledger } = trainReceipted("tamper");
  const rows = ledger.rows.map((r) => ({ ...r, payload: { ...r.payload } }));
  const victim = rows[5];
  (victim.payload as Record<string, unknown>).result = 0.999999;
  const [ok, , why] = verifyChain(rows as Array<Record<string, unknown>>);
  assert.equal(ok, false);
  assert.equal(why, "ROW_HASH_MISMATCH");
});

test("receipt without uniform (or vice versa) throws — loud, not silent", () => {
  const e = new MockSemanticEmbedder();
  const ledger = new _DrawLedger("substrate-embedding", "half");
  const uniform = new SeededUniform("x");
  assert.throws(
    () => e.train(CORPUS, 5, 1, { receipt: { ledger, uniform: undefined as never } }),
    /BOTH ledger and uniform/,
  );
  assert.throws(
    () => e.train(CORPUS, 5, 1, { receipt: { ledger: undefined as never, uniform } }),
    /BOTH ledger and uniform/,
  );
});

test("empty corpus books a named REFUSED row, not a silent fallback", () => {
  const e = new MockSemanticEmbedder();
  const ledger = new _DrawLedger("substrate-embedding", "empty");
  e.train(["", "  "], 5, 1, { receipt: { ledger, uniform: new SeededUniform("z") } });
  const refused = ledger.rows.find((r) => r.op === "REFUSED");
  assert.ok(refused);
  assert.equal((refused!.payload as Record<string, unknown>).reason, "empty_corpus");
});

test("unreceipted train keeps prior behavior (no rows, no throw)", () => {
  const e = new MockSemanticEmbedder();
  e.train(CORPUS, 5, 1);
  const v = e.embed("boats");
  assert.equal(v.length, 1024);
});

test("SeededUniform streams are reproducible and independent", () => {
  const a = new SeededUniform("same");
  const b = new SeededUniform("same");
  const c = new SeededUniform("other");
  const sa = [a.next(), a.next(), a.next()];
  const sb = [b.next(), b.next(), b.next()];
  const sc = [c.next(), c.next(), c.next()];
  assert.deepEqual(sa, sb);
  assert.notDeepEqual(sa, sc);
  assert.ok(sa.every((x) => x >= 0 && x < 1));
});
