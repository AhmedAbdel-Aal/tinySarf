import {createAnalyze} from "./scheduler";
import {getRuntime} from "./runtime";
export const analyze=createAnalyze(getRuntime);
export type {AnalyzeOptions,MorphAnalysis,MorphFeatures,MorphSpan,MorphSpanType} from "./schema";
