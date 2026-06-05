import { describe, expect, it } from "vitest";
import {
  forecastCommonSense,
  mapDecisionToVerdict,
  mapVerdictToDecision,
  scoreCseVerdict
} from "../src/index.js";

describe("cse verdict alignment", () => {
  it("maps approve to ready and block to blocked", () => {
    expect(mapDecisionToVerdict("approve")).toBe("ready");
    expect(mapDecisionToVerdict("warn")).toBe("warn");
    expect(mapDecisionToVerdict("ask_human")).toBe("ask_human");
    expect(mapDecisionToVerdict("block")).toBe("blocked");
    expect(mapVerdictToDecision("ready")).toBe("approve");
    expect(mapVerdictToDecision("blocked")).toBe("block");
  });

  it("scores force push as blocked with a reason", () => {
    const result = scoreCseVerdict({
      action: "git push --force",
      goal: "deploy",
      state: { branch: "main", repo_dirty: true, tests_passing: false }
    });

    expect(result.verdict).toBe("blocked");
    expect(result.forecast.decision).toBe("block");
    expect(result.reason.length).toBeGreaterThan(10);
  });

  it("downgrades approve to warn when thresholds are not met", () => {
    const forecast = forecastCommonSense({
      action: "npm install stale-package",
      state: {
        package_deprecated: true,
        package_last_updated_years_ago: 6
      }
    });
    expect(forecast.decision).toBe("warn");

    const scored = scoreCseVerdict({
      action: "npm install stale-package",
      state: {
        package_deprecated: true,
        package_last_updated_years_ago: 6
      }
    });
    expect(scored.verdict).toBe("warn");
  });
});