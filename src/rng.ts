/**
 * VENDORED from SuperInstance/substrate-rng @ 835702540ece8cf659244f8801956304b566723f
 * (main, PR #1) — Xoshiro256** + SplitMix64 seeder + stringToSeed excerpt.
 * Byte-identical output. Do not edit here; fix upstream and re-vendor.
 */
export const RNG_SOURCE = "substrate-rng@835702540ece8cf659244f8801956304b566723f";

const ROTL = (x: bigint, k: bigint): bigint => {
  const n = 64n;
  return ((x << k) | (x >> (n - k))) & 0xFFFFFFFFFFFFFFFFn;
};

const MASK = 0xFFFFFFFFFFFFFFFFn;

/** Seed-time mixer. SplitMix64 — used to expand a single u64 seed
 *  into a 256-bit Xoshiro state. */
function splitmix64(state: bigint): bigint {
  let z = (state + 0x9E3779B97F4A7C15n) & MASK;
  z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK;
  z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK;
  return (z ^ (z >> 31n)) & MASK;
}

export class Xoshiro256 {
  private s0: bigint;
  private s1: bigint;
  private s2: bigint;
  private s3: bigint;

  constructor(seed: bigint | string) {
    const seedBig = typeof seed === 'string' ? stringToSeed(seed) : seed;
    // Expand the seed via SplitMix64 to fill 256 bits of state.
    this.s0 = splitmix64(seedBig);
    this.s1 = splitmix64(this.s0);
    this.s2 = splitmix64(this.s1);
    this.s3 = splitmix64(this.s2);
    // Xoshiro256** requires not all zero state.
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0n) {
      this.s0 = 1n;
    }
  }

  /** Generate the next u64 from the state. Advances state. */
  nextU64(): bigint {
    const result = ROTL((this.s1 * 5n) & MASK, 7n) * 9n & MASK;
    const t = (this.s1 << 17n) & MASK;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = ROTL(this.s3, 45n);
    return result;
  }

  /** Random float in [0, 1). */
  next(): number {
    // Top 53 bits for double precision
    return Number(this.nextU64() >> 11n) / 9007199254740992;
  }

  /** Random integer in [0, n). */
  nextInt(n: number): number {
    if (n <= 0) throw new Error('nextInt: n must be > 0');
    return Math.floor(this.next() * n);
  }

  /** Random integer in [lo, hi] inclusive. */
  nextRange(lo: number, hi: number): number {
    return lo + this.nextInt(hi - lo + 1);
  }

  /** Random float in [lo, hi). */
  nextFloatRange(lo: number, hi: number): number {
    return lo + this.next() * (hi - lo);
  }

  /** Sample k from n without replacement (Fisher-Yates). */
  sample(n: number, k: number): number[] {
    if (k > n) throw new Error('sample: k > n');
    const arr = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) {
      const j = i + this.nextInt(n - i);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr.slice(0, k);
  }

  /** Standard normal via Box-Muller. */
  nextGaussian(mu = 0, sigma = 1): number {
    let u1 = this.next();
    if (u1 < 1e-15) u1 = 1e-15;
    const u2 = this.next();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mu + sigma * z;
  }

  /** Random bytes of `len`. */
  nextBytes(len: number): Uint8Array {
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = Number(this.nextU64() & 0xFFn);
    return out;
  }

  /** Hex string of `n` bytes. */
  nextHex(n: number): string {
    return Array.from(this.nextBytes(n)).map(b => b.toString(16).padStart(2, '0')).join('');
  }

  /** Bernoulli trial with probability p. */
  nextBool(p = 0.5): boolean { return this.next() < p; }

  /** Pick a random element from an array. */
  pick<T>(arr: readonly T[]): T {
    if (arr.length === 0) throw new Error('pick: empty array');
    return arr[this.nextInt(arr.length)];
  }

  /** Shuffle an array in place (Fisher-Yates). */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.nextInt(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  /** Serialize state to 4 u64s as a hex string. */
  serialize(): string {
    return [this.s0, this.s1, this.s2, this.s3].map(n => n.toString(16).padStart(16, '0')).join(':');
  }

  /** Restore from a serialized state. */
  static deserialize(s: string): Xoshiro256 {
    const parts = s.split(':').map(p => BigInt('0x' + p));
    const rng = Object.create(Xoshiro256.prototype) as Xoshiro256;
    rng.s0 = parts[0]; rng.s1 = parts[1]; rng.s2 = parts[2]; rng.s3 = parts[3];
    return rng;
  }
}

/**
 * PCG64 — Permuted Congruential Generator, 128-bit state.
 * Period 2^128. O'Neill 2014.
 */

export function stringToSeed(s: string): bigint {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    h = ((h ^ BigInt(c & 0xff)) * prime) & MASK;
    h = ((h ^ BigInt((c >> 8) & 0xff)) * prime) & MASK;
  }
  return h;
}

/** Convert a Date / timestamp / counter to a 64-bit seed. */
