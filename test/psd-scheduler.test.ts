import { describe, expect, it } from "vitest";
import { runPsdScheduler } from "../src/psd-scheduler.js";

describe("runPsdScheduler", () => {
  it("tracks drafted, ready, blocked, wasted, and prefetch metrics without GPU", async () => {
    const result = await runPsdScheduler({
      imagination: false,
      parallelPrefetch: false,
      current: {
        action: "run tests",
        goal: "ship feature",
        state: {
          branch: "feature/psd",
          repo_dirty: false,
          tests_passing: true
        }
      },
      branches: [
        { action: "inspect auth test failure", label: "Investigate" },
        {
          action: "git push --force",
          label: "Bad path",
          source: "agent_draft",
          state: { branch: "main", repo_dirty: true, tests_passing: false }
        }
      ]
    });

    expect(result.metrics.draftedBranches).toBe(2);
    expect(result.metrics.readyBranches).toBeGreaterThanOrEqual(1);
    expect(result.metrics.blockedBranches).toBeGreaterThanOrEqual(1);
    expect(result.metrics.wastedBranches).toBe(result.metrics.blockedBranches);
    expect(result.metrics.latencyMsCse).toHaveLength(2);
    expect(result.decoding.recommendedNext?.readiness).toBe("ready");
    expect(result.verified.map((item) => item.verdict)).toContain("blocked");
  });

  it("records prefetch hits when verifier finishes before local CSE", async () => {
    const fetchMock: typeof fetch = async (_input, init) => {
      const body = init?.body ? JSON.parse(String(init.body)) : {};
      const isVerifier = !body.response_format;
      if (isVerifier) {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return new Response(
          JSON.stringify({ choices: [{ message: { content: "looks executable" } }] }),
          { status: 200 }
        );
      }
      return new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: JSON.stringify({
                  branches: [{ action: "inspect failure" }, { action: "git push --force" }]
                })
              }
            }
          ]
        }),
        { status: 200 }
      );
    };

    const result = await runPsdScheduler({
      imagination: { fetch: fetchMock, timeoutMs: 2000 },
      parallelPrefetch: true,
      current: {
        action: "run tests",
        goal: "ship",
        state: { tests_passing: true, branch: "feature/psd" }
      }
    });

    expect(result.metrics.draftedBranches).toBe(2);
    expect(result.metrics.prefetchAttempts).toBeGreaterThanOrEqual(1);
    expect(result.metrics.latencyMsDraft.length).toBeGreaterThanOrEqual(1);
  });
});