import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { describe, expect, it } from "vitest";
import { checkVllmHealth, createVastMcpServer, extractVastConnectionInfo, listVastInstances, showVastInstance } from "../src/index.js";

type ToolContent = Array<{ type?: string; text?: string }>;

describe("Vast.ai MCP helpers", () => {
  it("lists and shows instances through the Vast API when an API key is provided", async () => {
    const fetchCalls: Array<{ headers?: HeadersInit; url: string }> = [];
    const fetchMock = async (input: string | URL, init?: RequestInit) => {
      fetchCalls.push({ headers: init?.headers, url: String(input) });
      const payload = String(input).endsWith("/api/v1/instances/")
        ? { instances: [{ id: 123, actual_status: "running" }] }
        : { instances: { id: 123, actual_status: "running", ssh_host: "203.0.113.10", ssh_port: 22022 } };
      return new Response(JSON.stringify(payload), { status: 200 });
    };

    const options = { apiKey: "test-key", fetch: fetchMock };
    await expect(listVastInstances(options)).resolves.toEqual([{ id: 123, actual_status: "running" }]);
    await expect(showVastInstance("123", options)).resolves.toMatchObject({
      actual_status: "running",
      id: 123
    });
    expect(fetchCalls[0]?.headers).toMatchObject({ Authorization: "Bearer test-key" });
  });

  it("falls back to the Vast CLI when no API key is configured", async () => {
    const argsSeen: string[][] = [];
    const runCli = async (args: string[]) => {
      argsSeen.push(args);
      return JSON.stringify({ instances: [{ id: 99, actual_status: "running" }] });
    };

    await expect(listVastInstances({ runCli })).resolves.toEqual([{ id: 99, actual_status: "running" }]);
    expect(argsSeen[0]).toEqual(["show", "instances", "--raw"]);
  });

  it("extracts SSH and vLLM API candidates from a Vast instance record", () => {
    const connection = extractVastConnectionInfo({
      actual_status: "running",
      id: 123,
      ports: {
        "8000/tcp": [{ HostIp: "198.51.100.2", HostPort: "18000" }],
        "8080/tcp": [{ HostIp: "198.51.100.2", HostPort: "18080" }]
      },
      ssh_host: "198.51.100.2",
      ssh_port: 22022
    });

    expect(connection.sshCommand).toBe("ssh -p 22022 root@198.51.100.2");
    expect(connection.apiCandidates).toEqual(["http://198.51.100.2:18000"]);
    expect(connection.ports.map((port) => port.containerPort)).toEqual([8000, 8080]);
  });

  it("checks vLLM health through /v1/models", async () => {
    const result = await checkVllmHealth("http://127.0.0.1:18000", {
      fetch: async (_input) =>
        new Response(JSON.stringify({ data: [{ id: "Qwen/Qwen2.5-14B-Instruct" }] }), {
          status: 200
        })
    });

    expect(result.ok).toBe(true);
    expect(result.url).toBe("http://127.0.0.1:18000/v1/models");
    expect(result.payload).toEqual({ data: [{ id: "Qwen/Qwen2.5-14B-Instruct" }] });
  });

  it("exposes read-only Vast tools over MCP", async () => {
    const server = await createVastMcpServer({
      apiKey: "test-key",
      fetch: async (input) => {
        if (String(input).includes("/api/v1/instances/")) {
          return new Response(JSON.stringify({ instances: [{ id: 123, actual_status: "running" }] }), { status: 200 });
        }
        return new Response(
          JSON.stringify({
            instances: {
              actual_status: "running",
              id: 123,
              ports: {
                "8000/tcp": [{ HostIp: "198.51.100.2", HostPort: "18000" }]
              },
              ssh_host: "198.51.100.2",
              ssh_port: 22022
            }
          }),
          { status: 200 }
        );
      },
      runCli: async () => "vastai help"
    });
    const client = new Client({ name: "vast-mcp-test-client", version: "1.0.0" });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);

    const tools = await client.listTools();
    expect(tools.tools.map((tool) => tool.name)).toEqual(
      expect.arrayContaining(["vast_status", "vast_list_instances", "vast_show_instance", "vast_connection_info", "vast_vllm_health"])
    );

    const listed = await client.callTool({ name: "vast_list_instances", arguments: {} });
    expect(JSON.parse(((listed.content as ToolContent)[0]?.text as string) ?? "[]")).toEqual([{ id: 123, actual_status: "running" }]);

    const connection = await client.callTool({ name: "vast_connection_info", arguments: { id: "123" } });
    expect(JSON.parse(((connection.content as ToolContent)[0]?.text as string) ?? "{}").apiCandidates).toEqual([
      "http://198.51.100.2:18000"
    ]);

    await client.close();
    await server.close();
  });
});
