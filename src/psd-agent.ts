import type { CommonSenseWorldState } from "./common-sense.js";
import { runPsdScheduler, type PsdSchedulerResult } from "./psd-scheduler.js";
import { createVastImaginationClient } from "./vast-client.js";

export interface PsdAgentStep {
  action: string;
  label?: string;
  state?: CommonSenseWorldState;
}

export interface PsdAgentInput {
  goal: string;
  steps: PsdAgentStep[];
  state?: CommonSenseWorldState;
  /** When true, uses local branches only (no GPU draft). */
  offline?: boolean;
}

export interface PsdAgentResult {
  scheduler: PsdSchedulerResult;
  executed?: string;
}

/**
 * Demo-style PSD agent entry: score a step queue with CSE while optionally drafting on GPU.
 * Pair with your own executor for bash/shell actions (same role as Python `agent.py`).
 */
export async function runPsdAgent(input: PsdAgentInput): Promise<PsdAgentResult> {
  const first = input.steps[0];
  if (!first) {
    throw new Error("PsdAgent requires at least one step.");
  }

  const scheduler = await runPsdScheduler({
    current: {
      action: first.action,
      goal: input.goal,
      state: { ...input.state, ...first.state }
    },
    goal: input.goal,
    state: input.state,
    branches: input.steps.slice(1).map((step) => ({
      action: step.action,
      label: step.label,
      state: step.state
    })),
    imagination: input.offline ? false : undefined
  });

  const next = scheduler.decoding.recommendedNext;
  return {
    scheduler,
    executed: next && next.readiness === "ready" ? next.action : undefined
  };
}

export async function probeVastImaginationEndpoints() {
  const client = createVastImaginationClient();
  return client.checkHealth();
}

/** Live GPU end-to-end demo — drafts branches on 18000, verifies with local CSE + 18001 prefetch. */
export async function runPsdAgentDemo() {
  const health = await probeVastImaginationEndpoints();
  if (!health.ok) {
    throw new Error(
      `Vast imagination endpoints are not healthy. draft=${health.draft.ok} verifier=${health.verifier.ok}`
    );
  }

  const scheduler = await runPsdScheduler({
    current: {
      action: "run tests",
      goal: "fix broken TypeScript file with failing tests",
      state: { tests_passing: false }
    },
    goal: "fix broken TypeScript file with failing tests",
    state: { tests_passing: false },
    imagination: {},
    parallelPrefetch: true
  });

  const { metrics, decoding, verified } = scheduler;
  const recommendedNext = decoding.recommendedNext;
  return {
    health,
    draftedActions: verified.map((item) => item.action),
    metrics: {
      draftedBranches: metrics.draftedBranches,
      readyBranches: metrics.readyBranches,
      blockedBranches: metrics.blockedBranches,
      wastedBranches: metrics.wastedBranches,
      prefetchAttempts: metrics.prefetchAttempts,
      prefetchHits: metrics.prefetchHits,
      prefetchHitRate: metrics.prefetchHitRate,
      avgDraftLatencyMs: metrics.avgDraftLatencyMs,
      projectedSavingsPerStep: metrics.projectedSavingsPerStep,
      estimatedWaitSavedMs: metrics.estimatedWaitSavedMs,
      latencyMsDraft: metrics.latencyMsDraft,
      latencyMsCse: metrics.latencyMsCse
    },
    mode: decoding.mode,
    recommendedNext: recommendedNext?.action,
    recommendedReadiness: recommendedNext?.readiness,
    executed: recommendedNext?.readiness === "ready" ? recommendedNext.action : undefined
  };
}

const isMain =
  typeof process !== "undefined" &&
  Boolean(process.argv[1]) &&
  (process.argv[1].endsWith("psd-agent.ts") || process.argv[1].endsWith("psd-agent.js"));

if (isMain) {
  runPsdAgentDemo()
    .then((output) => {
      console.log(JSON.stringify(output, null, 2));
    })
    .catch((error) => {
      console.error(error instanceof Error ? error.message : String(error));
      process.exitCode = 1;
    });
}