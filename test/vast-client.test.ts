import { describe, expect, it } from "vitest";
import { checkVastImaginationHealth, createVastImaginationClient, draftBranches } from "../src/vast-client.js";

describe("vast imagination client", () => {
  it("parses JSON branch drafts from chat completions", async () => {
    const fetchMock: typeof fetch = async () =>
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: JSON.stringify({
                  branches: [
                    { action: "inspect auth test failure", probability: 0.8 },
                    { action: "git push --force", probability: 0.1 }
                  ]
                })
              }
            }
          ]
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );

    const client = createVastImaginationClient({ fetch: fetchMock, timeoutMs: 1000 });
    const result = await client.draftBranches({
      goal: "fix tests",
      state: { tests_passing: false },
      currentAction: "run tests"
    });

    expect(result.branches).toHaveLength(2);
    expect(result.branches[0]?.action).toContain("inspect");
    expect(result.latencyMs).toBeGreaterThanOrEqual(0);
  });

  it("checks draft and verifier health in parallel", async () => {
    const fetchMock: typeof fetch = async () =>
      new Response(JSON.stringify({ data: [{ id: "m1" }] }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      });

    const health = await checkVastImaginationHealth({
      draftBaseUrl: "http://127.0.0.1:18000/v1",
      verifierBaseUrl: "http://127.0.0.1:18001/v1",
      fetch: fetchMock,
      timeoutMs: 1000
    });

    expect(health.ok).toBe(true);
    expect(health.draft.ok).toBe(true);
    expect(health.verifier.ok).toBe(true);
  });

  it("falls back to line parsing when model returns plain text", async () => {
    const result = await draftBranches(
      { goal: "demo", maxBranches: 3 },
      {
        draftBaseUrl: "http://127.0.0.1:18000/v1",
        model: "test-model",
        timeoutMs: 1000,
        fetch: async () =>
          new Response(
            JSON.stringify({
              choices: [{ message: { content: "- inspect failing test\n- run pytest -q\n" } }]
            }),
            { status: 200 }
          )
      }
    );

    expect(result.branches.length).toBeGreaterThanOrEqual(2);
  });
});