import { type CseThresholds, type CseVerdict, scoreCseVerdict } from "./cse-verdict.js";
import type { CommonSenseInput, CommonSenseWorldState } from "./common-sense.js";
import {
  type PsdBranch,
  type PsdBranchSource,
  type PredictiveSafetyDecodingInput,
  type PredictiveSafetyDecodingResult,
  runPredictiveSafetyDecoding
} from "./predictive-safety-decoding.js";
import {
  createVastImaginationClient,
  type VastImaginationBranch,
  type VastImaginationClientOptions
} from "./vast-client.js";

export interface PsdSchedulerMetrics {
  draftedBranches: number;
  readyBranches: number;
  blockedBranches: number;
  wastedBranches: number;
  latencyMsDraft: number[];
  latencyMsCse: number[];
  /** Sum of draft latency hidden per prefetch hit (full draft cost avoided on the next step). */
  estimatedWaitSavedMs: number;
  avgDraftLatencyMs: number;
  /** Expected draft latency hidden per scheduler step: prefetchHitRate × avgDraftLatencyMs. */
  projectedSavingsPerStep: number;
  prefetchAttempts: number;
  prefetchHits: number;
  prefetchHitRate: number;
}

export interface PsdSchedulerBranchResult {
  action: string;
  index: number;
  label?: string;
  latencyMsCse: number;
  prefetchLatencyMs?: number;
  prefetchHit: boolean;
  source: PsdBranchSource;
  verdict: CseVerdict;
  reason: string;
}

export interface PsdSchedulerInput {
  current: CommonSenseInput;
  goal?: string;
  state?: CommonSenseWorldState;
  /** Pre-seeded branches; when omitted and imagination is enabled, branches are drafted on GPU. */
  branches?: Array<{
    action: string;
    label?: string;
    probability?: number;
    source?: PsdBranchSource;
    state?: CommonSenseWorldState;
  }>;
  imagination?: VastImaginationClientOptions | false;
  thresholds?: Partial<CseThresholds>;
  maxBranches?: number;
  /** Run local CSE verification while prefetching the next branch on GPU. */
  parallelPrefetch?: boolean;
}

export interface PsdSchedulerResult {
  decoding: PredictiveSafetyDecodingResult;
  metrics: PsdSchedulerMetrics;
  verified: PsdSchedulerBranchResult[];
}

export function createEmptyPsdSchedulerMetrics(): PsdSchedulerMetrics {
  return {
    draftedBranches: 0,
    readyBranches: 0,
    blockedBranches: 0,
    wastedBranches: 0,
    latencyMsDraft: [],
    latencyMsCse: [],
    estimatedWaitSavedMs: 0,
    avgDraftLatencyMs: 0,
    projectedSavingsPerStep: 0,
    prefetchAttempts: 0,
    prefetchHits: 0,
    prefetchHitRate: 0
  };
}

export async function runPsdScheduler(input: PsdSchedulerInput): Promise<PsdSchedulerResult> {
  const metrics = createEmptyPsdSchedulerMetrics();
  const goal = input.goal ?? input.current.goal ?? "complete task";
  const state = input.state ?? input.current.state ?? {};
  const imaginationEnabled = input.imagination !== false;
  const imagination = imaginationEnabled ? createVastImaginationClient(input.imagination === false ? {} : input.imagination) : undefined;

  let drafted: Array<VastImaginationBranch & { state?: CommonSenseWorldState }> =
    input.branches?.map((branch) => ({
      action: branch.action,
      label: branch.label,
      probability: branch.probability,
      state: branch.state
    })) ?? [];

  if (drafted.length === 0 && imagination) {
    const draft = await imagination.draftBranches({
      goal,
      state,
      currentAction: normalizeCurrentAction(input.current),
      maxBranches: input.maxBranches ?? 4
    });
    metrics.latencyMsDraft.push(draft.latencyMs);
    drafted = draft.branches;
  }

  metrics.draftedBranches = drafted.length;

  const lastDraftLatencyMs = metrics.latencyMsDraft.at(-1) ?? 0;

  const verified = await verifyBranchesWithPrefetch({
    branches: drafted,
    goal,
    state,
    imagination,
    parallelPrefetch: input.parallelPrefetch !== false,
    thresholds: input.thresholds,
    metrics,
    lastDraftLatencyMs
  });

  for (const item of verified) {
    if (item.verdict === "ready") {
      metrics.readyBranches += 1;
    } else if (item.verdict === "blocked") {
      metrics.blockedBranches += 1;
      metrics.wastedBranches += 1;
    }
  }

  metrics.prefetchHitRate = metrics.prefetchAttempts > 0 ? round3(metrics.prefetchHits / metrics.prefetchAttempts) : 0;
  metrics.avgDraftLatencyMs = average(metrics.latencyMsDraft);
  metrics.projectedSavingsPerStep = round0(metrics.prefetchHitRate * metrics.avgDraftLatencyMs);

  const decoding = runPredictiveSafetyDecoding(buildDecodingInput(input, verified, metrics));

  return { decoding, metrics, verified };
}

async function verifyBranchesWithPrefetch(args: {
  branches: Array<VastImaginationBranch & { state?: CommonSenseWorldState }>;
  goal: string;
  state: CommonSenseWorldState;
  imagination?: ReturnType<typeof createVastImaginationClient>;
  parallelPrefetch: boolean;
  thresholds?: Partial<CseThresholds>;
  metrics: PsdSchedulerMetrics;
  lastDraftLatencyMs: number;
}): Promise<PsdSchedulerBranchResult[]> {
  const results: PsdSchedulerBranchResult[] = [];
  let prefetchPromise: Promise<{ action: string; latencyMs: number } | undefined> | undefined;

  for (let index = 0; index < args.branches.length; index += 1) {
    const branch = args.branches[index]!;
    const next = args.branches[index + 1];

    if (args.parallelPrefetch && args.imagination && next && !prefetchPromise) {
      args.metrics.prefetchAttempts += 1;
      prefetchPromise = args.imagination
        .verifyBranch({ action: next.action, goal: args.goal, state: args.state })
        .then((result) => ({ action: result.action, latencyMs: result.latencyMs }))
        .catch(() => undefined);
    }

    const cseStarted = Date.now();
    const scored = scoreCseVerdict(
      {
        action: branch.action,
        goal: args.goal,
        state: branch.state ?? args.state
      },
      args.thresholds
    );
    const cseEnded = Date.now();
    const latencyMsCse = cseEnded - cseStarted;
    args.metrics.latencyMsCse.push(latencyMsCse);

    let prefetchHit = false;
    let prefetchLatencyMs: number | undefined;

    if (prefetchPromise) {
      const prefetch = await prefetchPromise;
      prefetchPromise = undefined;
      const waitAfterCseMs = Date.now() - cseEnded;
      if (prefetch && prefetch.action === next?.action) {
        prefetchLatencyMs = prefetch.latencyMs;
        // Hit when verifier work overlapped CSE: we did not pay full sequential verify latency after CSE.
        prefetchHit = waitAfterCseMs < prefetch.latencyMs;
        if (prefetchHit) {
          args.metrics.prefetchHits += 1;
          // Next-step draft was produced during this step's window — credit full draft latency.
          args.metrics.estimatedWaitSavedMs += args.lastDraftLatencyMs;
        }
      }
    }

    results.push({
      action: branch.action,
      index,
      label: branch.label,
      latencyMsCse,
      prefetchHit,
      prefetchLatencyMs,
      source: "agent_draft",
      verdict: scored.verdict,
      reason: scored.reason
    });
  }

  return results;
}

function buildDecodingInput(
  input: PsdSchedulerInput,
  verified: PsdSchedulerBranchResult[],
  metrics: PsdSchedulerMetrics
): PredictiveSafetyDecodingInput {
  return {
    current: input.current,
    maxBranches: input.maxBranches ?? 6,
    maxRiskForReady: input.thresholds?.maxRiskForReady,
    minReversibilityForReady: input.thresholds?.minReversibilityForReady,
    minSuccessForReady: input.thresholds?.minSuccessForReady,
    candidates: verified.map((branch, index) => ({
      action: branch.action,
      label: branch.label,
      probability: branch.verdict === "ready" ? 0.82 : branch.verdict === "warn" ? 0.58 : 0.32,
      source: branch.source,
      state: input.branches?.[index]?.state ?? input.state ?? input.current.state
    }))
  };
}

function normalizeCurrentAction(current: CommonSenseInput) {
  if (typeof current.action === "string") {
    return current.action;
  }
  return [current.action.tool, current.action.command, current.action.description].filter(Boolean).join(" ");
}

function round3(value: number) {
  return Math.round(value * 1000) / 1000;
}

function round0(value: number) {
  return Math.round(value);
}

function average(values: number[]) {
  if (values.length === 0) {
    return 0;
  }
  return round0(values.reduce((sum, value) => sum + value, 0) / values.length);
}

export function summarizePsdSchedulerResult(result: PsdSchedulerResult): string {
  const { metrics, decoding } = result;
  return [
    `drafted_branches: ${metrics.draftedBranches}`,
    `ready_branches: ${metrics.readyBranches}`,
    `blocked_branches: ${metrics.blockedBranches}`,
    `wasted_branches: ${metrics.wastedBranches}`,
    `prefetch_hit_rate: ${metrics.prefetchHitRate}`,
    `avg_draft_latency_ms: ${metrics.avgDraftLatencyMs}`,
    `projected_savings_per_step_ms: ${metrics.projectedSavingsPerStep}`,
    `estimated_wait_saved_ms: ${metrics.estimatedWaitSavedMs}`,
    `latency_ms_draft: ${metrics.latencyMsDraft.join(", ") || "n/a"}`,
    `latency_ms_cse: ${metrics.latencyMsCse.join(", ") || "n/a"}`,
    `psd_mode: ${decoding.mode}`,
    decoding.recommendedNext ? `recommended_next: ${decoding.recommendedNext.action}` : "recommended_next: none"
  ].join("\n");
}