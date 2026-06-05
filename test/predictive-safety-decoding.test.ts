import { describe, expect, it } from "vitest";
import { runPredictiveSafetyDecoding, summarizePredictiveSafetyDecoding } from "../src/index.js";

describe("runPredictiveSafetyDecoding", () => {
  it("prepares the next safe branch while the current action is verified", () => {
    const result = runPredictiveSafetyDecoding({
      candidates: [
        {
          action: "inspect auth test failure",
          label: "Investigate likely failure",
          probability: 0.8
        },
        {
          action: "git push --force",
          label: "Bad branch",
          probability: 0.2,
          state: {
            branch: "main"
          }
        }
      ],
      current: {
        action: "run tests",
        goal: "ship feature",
        state: {
          branch: "feature/common-sense",
          repo_dirty: false,
          tests_passing: true
        }
      }
    });

    expect(result.mode).toBe("commit_and_serve");
    expect(result.recommendedNext?.action).toBe("inspect auth test failure");
    expect(result.recommendedNext?.readiness).toBe("ready");
    expect(result.blockedBranches.map((branch) => branch.action)).toContain("git push --force");
    expect(result.wallClockStrategy.join(" ")).toContain("pre-forecast in parallel");
  });

  it("replaces a blocked current action with a safer alternative branch", () => {
    const result = runPredictiveSafetyDecoding({
      current: {
        action: "git push --force",
        goal: "deploy feature",
        state: {
          branch: "main",
          repo_dirty: true,
          tests_passing: false
        }
      }
    });

    expect(result.current.decision).toBe("block");
    expect(result.mode).toBe("replace_current");
    expect(result.recommendedNext?.source).toBe("safer_alternative");
    expect(result.recommendedNext?.action).toContain("Commit, stash");
  });

  it("holds speculative branches that need review instead of treating them as ready", () => {
    const result = runPredictiveSafetyDecoding({
      candidates: [
        {
          action: "npm install stale-package",
          probability: 0.74,
          state: {
            package_deprecated: true,
            package_last_updated_years_ago: 6
          }
        }
      ],
      current: {
        action: "draft implementation plan",
        goal: "add markdown rendering",
        state: {}
      }
    });

    expect(result.mode).toBe("verify_current_first");
    expect(result.readyBranches).toHaveLength(0);
    expect(result.heldBranches[0]?.readiness).toBe("warn");
    expect(result.heldBranches[0]?.forecast.risks.map((risk) => risk.id)).toContain("stale_dependency");
  });

  it("renders a compact PSD summary", () => {
    const result = runPredictiveSafetyDecoding({
      candidates: [{ action: "inspect failing tests" }],
      current: {
        action: "run tests",
        state: {
          tests_passing: true
        }
      }
    });
    const summary = summarizePredictiveSafetyDecoding(result);

    expect(summary).toContain("PSD mode:");
    expect(summary).toContain("Ready branches:");
    expect(summary).toContain("Wall-clock strategy:");
  });
});
