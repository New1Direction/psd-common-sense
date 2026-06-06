import { type CseThresholds, type CseVerdict, resolveCseThresholds, scoreCseVerdict } from "./cse-verdict.js";
import {
  type CommonSenseAction,
  type CommonSenseForecast,
  type CommonSenseInput,
  type CommonSenseWorldState,
  forecastCommonSense
} from "./common-sense.js";

export type PsdBranchSource = "agent_draft" | "human_seed" | "memory" | "safer_alternative";

/** @deprecated Use `CseVerdict` from `./cse-verdict.js`. */
export type PsdBranchReadiness = CseVerdict;

export type PsdMode = "commit_and_serve" | "verify_current_first" | "replace_current" | "stop";

export interface PsdCandidate {
  action: CommonSenseAction;
  commitIf?: string[];
  goal?: string;
  label?: string;
  probability?: number;
  source?: PsdBranchSource;
  state?: CommonSenseWorldState;
}

export interface PredictiveSafetyDecodingInput {
  candidates?: PsdCandidate[];
  current: CommonSenseInput;
  includeSaferAlternatives?: boolean;
  maxBranches?: number;
  maxRiskForReady?: number;
  minReversibilityForReady?: number;
  minSuccessForReady?: number;
}

export interface PsdBranch {
  action: string;
  commitIf: string[];
  forecast: CommonSenseForecast;
  id: string;
  label: string;
  predictedUtility: number;
  readiness: PsdBranchReadiness;
  reason: string;
  source: PsdBranchSource;
}

export interface PredictiveSafetyDecodingResult {
  blockedBranches: PsdBranch[];
  current: CommonSenseForecast;
  discardedBranches: PsdBranch[];
  heldBranches: PsdBranch[];
  mode: PsdMode;
  recommendedNext?: PsdBranch;
  readyBranches: PsdBranch[];
  wallClockStrategy: string[];
}

type PsdThresholds = CseThresholds;

export function runPredictiveSafetyDecoding(input: PredictiveSafetyDecodingInput): PredictiveSafetyDecodingResult {
  const thresholds = resolveThresholds(input);
  const current = forecastCommonSense(input.current);
  const candidates = collectCandidates(input, current);
  const branches = candidates.map((candidate, index) => forecastCandidateBranch(input.current, candidate, index, thresholds));
  const ranked = [...branches].sort((a, b) => b.predictedUtility - a.predictedUtility);
  const readyBranches = ranked.filter((branch) => branch.readiness === "ready");
  const heldBranches = ranked.filter((branch) => branch.readiness === "warn" || branch.readiness === "ask_human");
  const blockedBranches = ranked.filter((branch) => branch.readiness === "blocked");
  const discardedBranches = blockedBranches.filter((branch) => branch.source !== "safer_alternative");
  const recommendedNext = chooseRecommendedBranch(current, readyBranches, heldBranches);
  const mode = chooseMode(current, recommendedNext);

  return {
    blockedBranches,
    current,
    discardedBranches,
    heldBranches,
    mode,
    recommendedNext,
    readyBranches,
    wallClockStrategy: describeWallClockStrategy(current, recommendedNext, heldBranches, blockedBranches)
  };
}

export function summarizePredictiveSafetyDecoding(result: PredictiveSafetyDecodingResult): string {
  const lines = [
    `PSD mode: ${result.mode}`,
    `Current action: ${result.current.transition.action}`,
    `Current decision: ${result.current.decision}`,
    `Ready branches: ${result.readyBranches.length}`,
    `Held branches: ${result.heldBranches.length}`,
    `Blocked branches: ${result.blockedBranches.length}`
  ];

  if (result.recommendedNext) {
    lines.push(
      `Recommended next: ${result.recommendedNext.action}`,
      `Why: ${result.recommendedNext.reason}`,
      `Utility: ${result.recommendedNext.predictedUtility.toFixed(2)}`
    );
  }

  if (result.wallClockStrategy.length > 0) {
    lines.push("Wall-clock strategy:", ...result.wallClockStrategy.map((item) => `- ${item}`));
  }

  return `${lines.join("\n")}\n`;
}

function resolveThresholds(input: PredictiveSafetyDecodingInput): PsdThresholds {
  return resolveCseThresholds({
    maxRiskForReady: input.maxRiskForReady,
    minReversibilityForReady: input.minReversibilityForReady,
    minSuccessForReady: input.minSuccessForReady
  });
}

function collectCandidates(input: PredictiveSafetyDecodingInput, current: CommonSenseForecast): PsdCandidate[] {
  const candidates = [...(input.candidates ?? [])];

  if (input.includeSaferAlternatives !== false) {
    for (const alternative of current.saferAlternatives) {
      candidates.push({
        action: alternative.action,
        commitIf: ["The current action is blocked or rejected.", "The observed world state still matches the forecast assumptions."],
        goal: input.current.goal,
        label: "Safer alternative",
        probability: Math.min(0.92, Math.max(0.45, Math.abs(alternative.expectedRiskDelta) + 0.58)),
        source: "safer_alternative",
        state: input.current.state
      });
    }
  }

  return dedupeCandidates(candidates).slice(0, input.maxBranches ?? 6);
}

function dedupeCandidates(candidates: PsdCandidate[]) {
  const seen = new Set<string>();
  const deduped: PsdCandidate[] = [];

  for (const candidate of candidates) {
    const key = normalizeAction(candidate.action);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(candidate);
  }

  return deduped;
}

function forecastCandidateBranch(current: CommonSenseInput, candidate: PsdCandidate, index: number, thresholds: PsdThresholds): PsdBranch {
  const scored = scoreCseVerdict(
    {
      action: candidate.action,
      goal: candidate.goal ?? current.goal,
      state: candidate.state ?? current.state
    },
    thresholds
  );
  const forecast = scored.forecast;
  const probability = clamp01(candidate.probability ?? 0.62);
  const predictedUtility = round2(probability * forecast.successProbability * (1 - forecast.risk) * forecast.reversibility);

  return {
    action: forecast.transition.action,
    commitIf: candidate.commitIf ?? defaultCommitConditions(),
    forecast,
    id: `psd-${index + 1}`,
    label: candidate.label ?? `Speculative branch ${index + 1}`,
    predictedUtility,
    readiness: scored.verdict,
    reason: scored.reason,
    source: candidate.source ?? "agent_draft"
  };
}

function chooseRecommendedBranch(
  current: CommonSenseForecast,
  readyBranches: PsdBranch[],
  heldBranches: PsdBranch[]
): PsdBranch | undefined {
  if (readyBranches.length > 0) {
    return readyBranches[0];
  }
  if (current.decision === "block") {
    return heldBranches.find((branch) => branch.source === "safer_alternative") ?? heldBranches[0];
  }
  return undefined;
}

function chooseMode(current: CommonSenseForecast, recommendedNext: PsdBranch | undefined): PsdMode {
  if (current.decision === "block") {
    return recommendedNext ? "replace_current" : "stop";
  }
  if (!recommendedNext) {
    return "verify_current_first";
  }
  return "commit_and_serve";
}

function describeWallClockStrategy(
  current: CommonSenseForecast,
  recommendedNext: PsdBranch | undefined,
  heldBranches: PsdBranch[],
  blockedBranches: PsdBranch[]
) {
  const strategy: string[] = [];

  if (current.decision === "block") {
    strategy.push("Do not execute the current action. Use the best safer branch as the replacement path.");
  } else {
    strategy.push("Verify or execute the current action while speculative branches are pre-forecast in parallel.");
  }

  if (recommendedNext) {
    strategy.push(`Commit and serve "${recommendedNext.action}" only after current verification clears and commit conditions still hold.`);
  }

  if (heldBranches.length > 0) {
    strategy.push("Keep warning or human-review branches prepared, but do not auto-commit them.");
  }

  if (blockedBranches.length > 0) {
    strategy.push("Discard blocked speculative branches; they are equivalent to failed bonus-token predictions.");
  }

  return strategy;
}

function defaultCommitConditions() {
  return [
    "The current action completed successfully.",
    "No new higher-risk world-state signal appeared.",
    "The user's goal has not changed."
  ];
}

function normalizeAction(action: CommonSenseAction): string {
  if (typeof action === "string") {
    return action.trim().toLowerCase().replace(/\s+/g, " ");
  }

  return [action.tool, action.command, action.description, action.target, JSON.stringify(action.parameters ?? {})]
    .filter((part): part is string => typeof part === "string" && part.length > 0)
    .join(" ")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function round2(value: number) {
  return Math.round(value * 100) / 100;
}
