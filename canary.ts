/**
 * Fleet canary (Issue #16): fnv1a-64("café Δ 日本語") === 0x024a555471370b18d.
 *
 * Every published substrate ship pins this value. The substrate's vectors
 * canary verifies that the FNV-1a basis is shared across the fleet.
 *
 * If this file's canary value disagrees with the computed hash, the ship
 * has drifted from the fleet.
 */

export const FLEET_CANARY_STRING = "café Δ 日本語";
export const FLEET_CANARY_VALUE = 0x024a555471370b18dn;

export function fnv1a64(s: string): bigint {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  const bytes = new TextEncoder().encode(s);
  for (const b of bytes) {
    h ^= BigInt(b);
    h = (h * prime) & mask;
  }
  return h;
}

export function verify_fleet_canary(): boolean {
  return fnv1a64(FLEET_CANARY_STRING) === BigInt(FLEET_CANARY_VALUE);
}
