/**
 * substrate-rng quilt ledger: receipted randomness.
 *
 * Every draw from a seedable RNG is a deterministic consequence of
 * (seed, algorithm, state, draw_index). This module books those
 * consequences as a hash-chained receipt ledger in the fleet's
 * 4quilt family envelope — the same canonical JSON + fnv1a-32 recipe
 * the Python ledgers (laya4quilt, tagseq2tagseq4quilt, gpu_bpe4quilt,
 * jev-ultrafast-quilt) speak. Rows verify cross-language: a chain
 * booked here verifies with the Python family verifier unmodified,
 * and vice versa (src/tests/ledger.test.ts freezes real Python-family
 * rows as fixtures and checks them here).
 *
 * Storage owns facts about records; producers own vocabularies. The
 * verifier checks envelope + canonical bytes + hashes + chain order,
 * never payload semantics.
 *
 * Canonical-JSON contract (matches Python json.dumps sort_keys=True,
 * separators=(",",":"), ensure_ascii=False bit-for-bit on the family
 * payload domain):
 *   - object keys sorted; no insignificant whitespace
 *   - non-ASCII emitted raw (UTF-8); C0 controls escaped as \uXXXX
 *   - numbers: shortest round-trip (JS String() == Python repr for
 *     non-integral doubles). Boundaries, avoided by construction:
 *     integral floats (JS "1" vs Python "1.0") and exponent padding
 *     (JS "1e-7" vs Python "1e-07"). Family payloads use integers for
 *     counts/indices and fractional floats for timestamps.
 *
 * Family field-convention drift (bound by hash, verifier-agnostic):
 *   tick base: 1-based here (gpu_bpe/tagseq convention; jev books
 *     0-based — its chain is self-consistent, don't rebase it).
 *   ts: fractional-ms float here (tagseq rounds to 6dp; gpu_bpe raw;
 *     jev uses ISO strings). The envelope binds whatever is present.
 */
/**
 * VENDORED from SuperInstance/substrate-rng @ 835702540ece8cf659244f8801956304b566723f
 * (main, PR #1). Byte-identical. Do not edit here; fix upstream and re-vendor.
 * The vendored Xoshiro256 below is the same upstream commit's src/index.ts
 * excerpt, kept so this package stays dependency-free while producing
 * byte-exact fleet RNG streams.
 */

import { createHash } from "node:crypto";

export const GENESIS = "0".repeat(8);
const FNV1A_OFFSET = 0x811c9dc5;
const FNV1A_PRIME = 0x01000193;

/** fnv1a-32 over UTF-8 bytes. Bit-identical to the Python family's
 *  fnv1a32(canonical(obj).encode("utf-8")). */
export function fnv1a32(data: Uint8Array | string): number {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
  let h = FNV1A_OFFSET;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, FNV1A_PRIME) >>> 0;
  }
  return h >>> 0;
}

/** Canonical JSON per the contract in the header. */
export function canonical(obj: unknown): string {
  if (obj === null) return "null";
  if (typeof obj === "boolean") return obj ? "true" : "false";
  if (typeof obj === "number") {
    if (!Number.isFinite(obj)) throw new Error("non-finite number in canonical payload");
    return String(obj);
  }
  if (typeof obj === "string") return quote(obj);
  if (Array.isArray(obj)) return "[" + obj.map((v) => canonical(v)).join(",") + "]";
  if (typeof obj === "object") {
    const rec = obj as Record<string, unknown>;
    const keys = Object.keys(rec).sort();
    return "{" + keys.map((k) => quote(k) + ":" + canonical(rec[k])).join(",") + "}";
  }
  throw new Error("unserializable value in canonical payload");
}

function quote(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 0x22) out += '\\"';
    else if (c === 0x5c) out += "\\\\";
    else if (c === 0x08) out += "\\b";
    else if (c === 0x09) out += "\\t";
    else if (c === 0x0a) out += "\\n";
    else if (c === 0x0c) out += "\\f";
    else if (c === 0x0d) out += "\\r";
    else if (c < 0x20) out += "\\u" + c.toString(16).padStart(4, "0");
    else out += s[i]; // ensure_ascii=False parity: non-ASCII raw
  }
  return out + '"';
}

export function sha256Hex(data: Uint8Array | string): string {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
  return createHash("sha256").update(bytes).digest("hex");
}

export const PRODUCER = { tool: "substrate-rng-ledger", vocabulary: "rng-run/v1" };
export const REQUIRED_ENVELOPE = ["tick", "ts", "op", "actor", "payload", "chain_prev", "row_hash"] as const;

export type RowOp = "BIND" | "EFFECT" | "REFUSED" | "TICK" | "VIEW";

export interface LedgerRow {
  tick: number;
  ts: number;
  op: RowOp;
  actor: string;
  payload: Record<string, unknown>;
  chain_prev: string;
  row_hash: string;
  draw_index?: number;
}

/** Fractional-ms timestamp, always non-integral (avoids the JS "1" vs
 *  Python "1.0" integral-float divergence in canonical form). */
const now = (): number => {
  const t = (Date.now() + (performance.now() % 1)) / 1000;
  return Number(t.toFixed(6));
};

export class DrawLedger {
  readonly actor: string;
  readonly run_id: string;
  rows: LedgerRow[] = [];
  head: string = GENESIS;

  constructor(actor: string, run_id?: string) {
    this.actor = actor;
    this.run_id = run_id ?? Math.random().toString(16).slice(2, 10);
  }

  private book(op: RowOp, payload: Record<string, unknown>, ctx: Record<string, unknown> = {}): LedgerRow {
    const row: Record<string, unknown> = {
      tick: this.rows.length + 1,
      ts: now(),
      op,
      actor: this.actor,
      payload,
      chain_prev: this.head,
      ...ctx,
    };
    const hash = hash8(row);
    const sealed = { ...row, row_hash: hash } as unknown as LedgerRow;
    this.rows.push(sealed);
    this.head = hash;
    return sealed;
  }

  /** BIND the stochastic process: seed + algorithm + initial state binding. */
  bind(algo: string, seed: string | number, serializedState: string, opts: Record<string, unknown> = {}): LedgerRow {
    return this.book("BIND", {
      producer: PRODUCER,
      kind: "rng-run/v1",
      algo,
      seed: String(seed),
      state_sha256: sha256Hex(serializedState),
      ...opts,
    });
  }

  /** EFFECT per draw: distribution + params + result + state bindings. */
  draw(
    drawIndex: number,
    distribution: string,
    params: Record<string, unknown>,
    result: number,
    stateBefore: string,
    stateAfter: string
  ): LedgerRow {
    return this.book(
      "EFFECT",
      {
        kind: "draw/v1",
        distribution,
        params,
        result,
        state_before_sha256: sha256Hex(stateBefore),
        state_after_sha256: sha256Hex(stateAfter),
      },
      { draw_index: drawIndex }
    );
  }

  /** TICK heartbeat for batch draws (e.g. shuffle progress). */
  heartbeat(drawIndex: number, note = ""): LedgerRow {
    return this.book("TICK", { kind: "heartbeat/v1", note }, { draw_index: drawIndex });
  }

  /** Named refusal: the process refused to continue. The canonical case:
   *  deserialize() then observe a state-hash mismatch — replay divergence
   *  is a lie about determinism, so it books a visible REFUSED row. */
  refuse(reason: string, detail: Record<string, unknown> = {}): LedgerRow {
    return this.book("REFUSED", { producer: PRODUCER, kind: "refusal/v1", reason, ...detail });
  }

  /** VIEW of exported artifacts; head captured BEFORE this row books. */
  view(artifacts: Array<{ path: string; sha256: string }>): LedgerRow {
    const headBefore = this.head;
    return this.book("VIEW", { kind: "exports/v1", head: headBefore, artifacts });
  }

  verify(): [boolean, string | null, string | null] {
    return verifyChain(this.rows);
  }

  exportRows(): LedgerRow[] {
    return this.rows;
  }

  canon(): Record<string, unknown> {
    return {
      format: "quilt-rng-run/v1",
      run_id: this.run_id,
      head: this.head,
      count: this.rows.length,
      rows: this.rows,
    };
  }
}

function hash8(row: Record<string, unknown>): string {
  return fnv1a32(canonical(row)).toString(16).padStart(8, "0");
}

/** Generic family verifier. Returns (ok, bad_row_hash|null, reason|null),
 *  the exact convention of the Python family. Never interprets payloads. */
export function verifyChain(rows: Array<Record<string, unknown>>): [boolean, string | null, string | null] {
  let prev: string = GENESIS;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const missing = REQUIRED_ENVELOPE.filter((f) => !(f in row));
    if (missing.length > 0) {
      return [false, (row.row_hash as string) ?? null, `MISSING_ENVELOPE_FIELDS:${missing.join(",")}`];
    }
    if (row.chain_prev !== prev) {
      return [false, (row.row_hash as string) ?? null, `CHAIN_BREAK_AT_ROW_${i}`];
    }
    const body = { ...row };
    delete body.row_hash;
    if (row.row_hash !== hash8(body)) {
      return [false, (row.row_hash as string) ?? null, "ROW_HASH_MISMATCH"];
    }
    prev = row.row_hash as string;
  }
  return [true, null, null];
}
