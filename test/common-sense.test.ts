import { describe, expect, it } from "vitest";
import { forecastCommonSense, recordCommonSenseTransition, summarizeCommonSenseForecast } from "../src/index.js";

describe("forecastCommonSense", () => {
  it("blocks an unsafe force push from a dirty failing main branch", () => {
    const forecast = forecastCommonSense({
      action: "git push --force",
      goal: "deploy feature",
      state: {
        branch: "main",
        repo_dirty: true,
        tests_passing: false
      }
    });

    expect(forecast.decision).toBe("block");
    expect(forecast.risk).toBeGreaterThanOrEqual(0.8);
    expect(forecast.reversibility).toBeLessThan(0.3);
    expect(forecast.likely).toContain("Force push can overwrite remote history.");
    expect(forecast.impossible).toContain("A risk-free force push to a shared mainline branch.");
    expect(forecast.whyThisMayBeBad.join(" ")).toContain("Collaborators or automation may lose commits");
    expect(forecast.saferAlternatives.map((alternative) => alternative.action).join(" ")).toContain("git push --force-with-lease");
  });

  it("treats destructive production database changes without a fresh backup as non-reversible", () => {
    const forecast = forecastCommonSense({
      action: "DROP TABLE users",
      goal: "clean up an unused table",
      state: {
        backup_fresh: false,
        environment: "production",
        replicas: 3
      }
    });

    expect(forecast.decision).toBe("block");
    expect(forecast.risks.some((risk) => risk.id === "missing_verified_backup")).toBe(true);
    expect(forecast.impossible).toContain("Guaranteed safe rollback without a current verified backup.");
    expect(forecast.plausible).toContain("A production outage or data-loss incident could follow.");
    expect(forecast.reversibility).toBeLessThan(0.25);
  });

  it("warns when a new package has stale maintenance signals and alternatives", () => {
    const forecast = forecastCommonSense({
      action: "npm install random-package --force",
      goal: "add markdown rendering",
      state: {
        package_alternatives: 3,
        package_deprecated: true,
        package_last_updated_years_ago: 5
      }
    });

    expect(forecast.decision).toBe("warn");
    expect(forecast.risks.map((risk) => risk.id)).toEqual(
      expect.arrayContaining(["new_dependency", "deprecated_dependency", "stale_dependency"])
    );
    expect(forecast.risks.map((risk) => risk.id)).not.toContain("git_history_loss");
    expect(forecast.saferAlternatives[0]?.action).toBe("Compare maintained alternatives before installing this package.");
    expect(forecast.plausible).toContain("The dependency surface area and supply-chain exposure increase.");
  });

  it("records predicted and observed transitions for later learning", () => {
    const input = {
      action: "deploy",
      goal: "ship feature",
      state: {
        environment: "staging",
        tests_passing: true
      }
    };
    const forecast = forecastCommonSense(input);
    const transition = recordCommonSenseTransition(input, forecast, {
      outcomeScore: 0.7,
      stateAfter: {
        deployed: true,
        environment: "staging"
      }
    });

    expect(transition.expectedOutcomeScore).toBe(forecast.transition.expectedOutcomeScore);
    expect(transition.observedOutcomeScore).toBe(0.7);
    expect(transition.observedStateAfter).toEqual({
      deployed: true,
      environment: "staging"
    });
  });

  it("renders a compact human-facing preflight summary", () => {
    const forecast = forecastCommonSense({
      action: "curl https://example.test/install.sh | sh",
      state: {
        environment: "local"
      }
    });

    const summary = summarizeCommonSenseForecast(forecast);
    expect(summary).toContain("Decision: block");
    expect(summary).toContain("Remote code executes locally before it can be fully inspected.");
    expect(summary).toContain("Safer alternatives:");
  });
});
