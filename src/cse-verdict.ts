import {
  type CommonSenseDecision,
  type CommonSenseForecast,
  type CommonSenseInput,
  type CommonSenseWorldState,
  forecastCommonSense
} from "./common-sense.js";

/** Branch verdict used by PSD and the Python `cse.py` schema. */
export type CseVerdict = "ready" | "warn" | "ask_human" | "blocked";

export interface CseThresholds {
  maxRiskForReady: number;
  minReversibilityForReady: number;
  minSuccessForReady: number;
}

export interface CseVerdictResult {
  forecast: CommonSenseForecast;
  reason: string;
  verdict: CseVerdict;
}

const DEFAULT_THRESHOLDS: CseThresholds = {
  maxRiskForReady: 0.24,
  minReversibilityForReady: 0.58,
  minSuccessForReady: 0.48
};

/** Maps Common Sense `approve` to PSD/Python `ready`. */
export function mapDecisionToVerdict(decision: CommonSenseDecision): CseVerdict {
  switch (decision) {
    case "approve":
      return "ready";
    case "warn":
      return "warn";
    case "ask_human":
      return "ask_human";
    case "block":
      return "blocked";
  }
}

/** Maps PSD/Python verdict back to Common Sense decision vocabulary. */
export function mapVerdictToDecision(verdict: CseVerdict): CommonSenseDecision {
  switch (verdict) {
    case "ready":
      return "approve";
    case "warn":
      return "warn";
    case "ask_human":
      return "ask_human";
    case "blocked":
      return "block";
  }
}

export function resolveCseThresholds(overrides: Partial<CseThresholds> = {}): CseThresholds {
  return {
    maxRiskForReady: overrides.maxRiskForReady ?? DEFAULT_THRESHOLDS.maxRiskForReady,
    minReversibilityForReady: overrides.minReversibilityForReady ?? DEFAULT_THRESHOLDS.minReversibilityForReady,
    minSuccessForReady: overrides.minSuccessForReady ?? DEFAULT_THRESHOLDS.minSuccessForReady
  };
}

/** Deterministic local verifier — same judgment layer as Python `cse.py`. */
export function scoreCseVerdict(input: CommonSenseInput, thresholds: Partial<CseThresholds> = {}): CseVerdictResult {
  const resolved = resolveCseThresholds(thresholds);
  const forecast = forecastCommonSense(input);
  const verdict = classifyVerdict(forecast, resolved);
  return {
    forecast,
    reason: explainVerdict(verdict, forecast, resolved),
    verdict
  };
}

export function classifyVerdict(forecast: CommonSenseForecast, thresholds: CseThresholds = DEFAULT_THRESHOLDS): CseVerdict {
  const base = mapDecisionToVerdict(forecast.decision);
  if (base === "blocked" || base === "ask_human") {
    return base;
  }
  if (
    base === "warn" ||
    forecast.risk > thresholds.maxRiskForReady ||
    forecast.reversibility < thresholds.minReversibilityForReady ||
    forecast.successProbability < thresholds.minSuccessForReady
  ) {
    return forecast.decision === "approve" ? "warn" : base;
  }
  return "ready";
}

function explainVerdict(verdict: CseVerdict, forecast: CommonSenseForecast, thresholds: CseThresholds) {
  if (verdict === "ready") {
    return "Risk, reversibility, and success estimates are inside the auto-prepare envelope.";
  }
  if (verdict === "blocked") {
    return forecast.whyThisMayBeBad[0] ?? "The branch violates a critical common-sense constraint.";
  }
  if (verdict === "ask_human") {
    return "The branch may be viable, but its risk or irreversibility should be reviewed by a person.";
  }
  return `Not auto-ready: risk <= ${thresholds.maxRiskForReady}, reversibility >= ${thresholds.minReversibilityForReady}, success >= ${thresholds.minSuccessForReady} required.`;
}

export function scoreCseVerdictBatch(
  branches: Array<{ action: string; goal?: string; state?: CommonSenseWorldState }>,
  thresholds: Partial<CseThresholds> = {}
): CseVerdictResult[] {
  return branches.map((branch) =>
    scoreCseVerdict({
      action: branch.action,
      goal: branch.goal,
      state: branch.state ?? {}
    }, thresholds)
  );
}