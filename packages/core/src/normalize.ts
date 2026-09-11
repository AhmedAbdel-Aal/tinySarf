import { NORMALIZATION_VERSION } from "./schema";
export const LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهويى";
export const MAX_LENGTH = 32;
export const MAX_BATCH = 2048;
const MARK = /^[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]$/u;
export interface NormalizedWord {
  original: string; text: string; offsets: { start: number; end: number }[];
  packed: Uint32Array; version: typeof NORMALIZATION_VERSION;
}
/** No spelling folding: hamza, alef maqsura and ta marbuta remain distinct. */
export function normalize(input: string): NormalizedWord {
  if (typeof input !== "string") throw new TypeError("Expected one Arabic word");
  if (!input.length || input.length > 512) throw new RangeError("Word must contain 1–32 Arabic letters (at most 512 code units)");
  const offsets: NormalizedWord["offsets"] = [], ids: number[] = [];
  let text = "", offset = 0;
  for (const ch of input) {
    if (MARK.test(ch) || ch === "ـ") {
      if (!offsets.length) throw new TypeError("Leading marks or tatweel are unsupported");
      offsets.at(-1)!.end = offset + ch.length;
      if (ch !== "ـ") ids[ids.length - 1] |= (1 << 16);
    } else {
      const id = LETTERS.indexOf(ch);
      if (id < 0) throw new TypeError("Expected Arabic letters only; whitespace, presentation forms and foreign text are unsupported");
      text += ch; ids.push(id + 1); offsets.push({ start: offset, end: offset + ch.length });
    }
    offset += ch.length;
  }
  if (!ids.length || ids.length > MAX_LENGTH) throw new RangeError(`Maximum word length is ${MAX_LENGTH} Arabic letters`);
  const packed = Uint32Array.from(ids, (id, i) => id | (i << 8) | ((i === 0 ? 1 : 0) << 17) | ((i === ids.length - 1 ? 1 : 0) << 18));
  return { original: input, text, offsets, packed, version: NORMALIZATION_VERSION };
}
export function sizeClass(length: number): number { return length <= 8 ? 8 : length <= 16 ? 16 : 32; }
