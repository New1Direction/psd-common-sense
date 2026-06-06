import { describe, expect, it } from "vitest";
import {
  type BenchStepResult,
  aggregate,
  createSimulatedImagination,
  formatSummaryTable,
  runBenchmark,
  toCsv
} from "../bench/bench.js";
import { parseArgs } from "../bench/run-bench.js";

function stepResult(overrides: Partial<BenchStepResult>): BenchStepResult {
  return {
    index: 0,
    action: "do a thing",
    draftedBranches: 4,
    readyBranches: 2,
    blockedBranches: 1,
    wastedBranches: 1,
    prefetchAttempts: 3,
    prefetchHits: 3,
    prefetchHitRate: 1,
    draftLatencyMs: 4000,
    avgCseLatencyMs: 1,
    estimatedWaitSavedMs: 12000,
    mode: "commit_and_serve",
    recommendedNext: "next",
    error: "",
    ...overrides
  };
}

describe("aggregate", () => {
  it("computes hit rate, latency percentiles, and failed steps over only successful steps", () => {
    // Arrange
    const results = [
      stepResult({ index: 0, draftLatencyMs: 4000, prefetchAttempts: 3, prefetchHits: 3, estimatedWaitSavedMs: 12000 }),
      stepResult({ index: 1, draftLatencyMs: 2000, prefetchAttempts: 3, prefetchHits: 0, estimatedWaitSavedMs: 0 }),
      stepResult({ index: 2, error: "boom" })
    ];

    // Act
    const summary = aggregate(results);

    // Assert
    expect(summary.steps).toBe(3);
    expect(summary.failedSteps).toBe(1);
    expect(summary.prefetchAttempts).toBe(6);
    expect(summary.prefetchHits).toBe(3);
    expect(summary.prefetchHitRate).toBe(0.5);
    expect(summary.avgDraftLatencyMs).toBe(3000);
    expect(summary.p50DraftLatencyMs).toBe(2000);
    expect(summary.p95DraftLatencyMs).toBe(4000);
    expect(summary.projectedSavingsPerStepMs).toBe(1500);
    expect(summary.totalEstimatedWaitSavedMs).toBe(12000);
  });

  it("returns zeroed rates when there are no successful steps", () => {
    const summary = aggregate([stepResult({ error: "x" })]);
    expect(summary.prefetchHitRate).toBe(0);
    expect(summary.avgDraftLatencyMs).toBe(0);
  });
});

describe("toCsv", () => {
  it("emits a header plus one row per step and quotes commas", () => {
    const csv = toCsv([stepResult({ action: "git add, then commit" })]);
    const lines = csv.trim().split("\n");
    expect(lines[0]).toContain("index,action");
    expect(lines).toHaveLength(2);
    expect(lines[1]).toContain('"git add, then commit"');
  });
});

describe("formatSummaryTable", () => {
  it("separates measured numbers from projections", () => {
    const table = formatSummaryTable(aggregate([stepResult({})]));
    expect(table).toContain("### Measured");
    expect(table).toContain("### Projected");
    expect(table).toContain("Prefetch hit rate");
  });
});

describe("runBenchmark (offline, simulated GPU)", () => {
  it("drives every step and produces measured per-step results without a real endpoint", async () => {
    // Arrange
    const trace = [{ action: "run tests" }, { action: "read the error" }, { action: "apply a fix" }];
    const imagination = createSimulatedImagination({ draftLatencyMs: 8, verifyLatencyMs: 2 });

    // Act
    const { results, summary } = await runBenchmark({ trace, imagination, parallelPrefetch: true });

    // Assert
    expect(results).toHaveLength(3);
    expect(summary.steps).toBe(3);
    expect(summary.failedSteps).toBe(0);
    expect(summary.draftedBranches).toBeGreaterThan(0);
    expect(summary.blockedBranches).toBeGreaterThanOrEqual(1); // the curl|bash branch is always blocked
    expect(summary.avgDraftLatencyMs).toBeGreaterThan(0);
  });

  it("records an error per step and keeps going when the GPU call fails", async () => {
    const failing = {
      fetch: (async () => {
        throw new Error("net down");
      }) as typeof fetch
    };

    const { results, summary } = await runBenchmark({
      trace: [{ action: "a" }, { action: "b" }],
      imagination: failing
    });

    expect(summary.steps).toBe(2);
    expect(summary.failedSteps).toBe(2);
    expect(results.every((result) => result.error !== "")).toBe(true);
  });
});

describe("parseArgs", () => {
  it("reads flags and falls back to defaults", () => {
    const options = parseArgs(["--offline", "--repeat", "2", "--trace", "/tmp/t.json", "--max-branches", "6"]);
    expect(options.offline).toBe(true);
    expect(options.repeat).toBe(2);
    expect(options.tracePath).toBe("/tmp/t.json");
    expect(options.maxBranches).toBe(6);
    expect(options.outPath).toBe("bench/results.csv");
  });
});
