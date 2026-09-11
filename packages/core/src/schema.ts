export const SCHEMA_VERSION = 1;
export const NORMALIZATION_VERSION = "arabic-v1";
export const SPAN_TYPES = ["conjunction", "particle", "preposition", "article", "stem", "derivational_suffix", "inflectional_suffix", "pronominal_enclitic"] as const;
export const POS_TYPES = ["noun","proper_noun","numeral","adjective","verb","adverb","preposition","conjunction","interjection","pronoun","particle","unknown"] as const;
export type MorphSpanType = (typeof SPAN_TYPES)[number];
export interface MorphSpan { type: MorphSpanType; start: number; end: number }
export interface MorphFeatures {
  person?: string; gender?: string; number?: string; aspect?: string;
  mood?: string; voice?: string; case?: string; state?: string;
}
export interface MorphAnalysis {
  spans: MorphSpan[]; root: string | null; pattern: string | null;
  lemma?: string | null; pos: string; features: MorphFeatures; score: number;
}
export interface AnalyzeOptions {
  topK?: number;
  /** WebGPU is the default. CPU is an explicit reference backend, never a silent fallback. */
  backend?: "webgpu" | "cpu";
  /** Average log ranking score threshold. Uncalibrated; default is no abstention. */
  threshold?: number;
}
export const FEATURE_VALUES = {
  person: ["first", "second", "third"], gender: ["masculine", "feminine"],
  number: ["singular", "dual", "plural"], aspect: ["perfective", "imperfective", "imperative"],
  mood: ["indicative", "subjunctive", "jussive"], voice: ["active", "passive"],
  case: ["nominative", "accusative", "genitive"], state: ["definite", "indefinite", "construct"],
} as const;
export function validateAnalysis(word: string, a: MorphAnalysis): void {
  if (!Number.isFinite(a.score) || !(POS_TYPES as readonly string[]).includes(a.pos)) throw new Error("Invalid score or POS");
  if (a.pattern!==null && (typeof a.pattern!=="string" || !a.pattern.length)) throw new Error("Invalid pattern");
  if (a.lemma!==undefined && a.lemma!==null && (typeof a.lemma!=="string" || !a.lemma.length)) throw new Error("Invalid lemma");
  if (a.root !== null && !/^[ء-غف-ي]{3,4}$/u.test(a.root)) throw new Error("Invalid root");
  let end = 0;
  for (const s of a.spans) {
    if (!SPAN_TYPES.includes(s.type) || !Number.isInteger(s.start) || !Number.isInteger(s.end) || s.start !== end || s.end <= s.start || s.end > word.length) throw new Error("Invalid span coverage");
    if (s.end<word.length && /^[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]$/u.test(word[s.end])) throw new Error("Span boundary splits an attached mark");
    end = s.end;
  }
  if (end !== word.length || !a.spans.some(s => s.type === "stem")) throw new Error("Missing word coverage or stem");
  for (const [key, value] of Object.entries(a.features)) {
    if (!(key in FEATURE_VALUES) || !(FEATURE_VALUES[key as keyof MorphFeatures] as readonly string[]).includes(value)) throw new Error(`Invalid feature ${key}`);
  }
}
