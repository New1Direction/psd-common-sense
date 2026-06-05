import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import { promisify } from "node:util";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const execFileAsync = promisify(execFile);
const DEFAULT_API_BASE_URL = "https://console.vast.ai";
const DEFAULT_CLI_PATH = "vastai";

type FetchLike = (input: string | URL, init?: RequestInit) => Promise<Response>;

export interface VastMcpOptions {
  apiBaseUrl?: string;
  apiKey?: string;
  cliPath?: string;
  fetch?: FetchLike;
  runCli?: (args: string[]) => Promise<string>;
}

export interface VastPortMapping {
  containerPort: number;
  host?: string;
  hostPort?: number;
  label?: string;
  url?: string;
}

export interface VastConnectionInfo {
  apiCandidates: string[];
  instanceId?: string | number;
  ports: VastPortMapping[];
  sshCommand?: string;
  sshHost?: string;
  sshPort?: number;
  status?: string;
}

export async function createVastMcpServer(options: VastMcpOptions = {}): Promise<McpServer> {
  const server = new McpServer({
    name: "vast-ai",
    version: "0.1.0",
    websiteUrl: "https://vast.ai"
  });

  server.registerTool(
    "vast_status",
    {
      description: "Check whether Vast.ai auth and CLI access are configured. Does not reveal secrets."
    },
    safeHandler(async () => asToolText(await getVastStatus(options)))
  );

  server.registerTool(
    "vast_list_instances",
    {
      description: "List current Vast.ai instances using VAST_API_KEY or the vastai CLI. Read-only."
    },
    safeHandler(async () => asToolText(await listVastInstances(options)))
  );

  server.registerTool(
    "vast_show_instance",
    {
      description: "Show one Vast.ai instance by id. Read-only.",
      inputSchema: {
        id: z.string().min(1).describe("Vast.ai instance id")
      }
    },
    safeHandler(async ({ id }) => asToolText(await showVastInstance(id, options)))
  );

  server.registerTool(
    "vast_connection_info",
    {
      description: "Extract SSH and likely vLLM API connection info from one Vast.ai instance record. Read-only.",
      inputSchema: {
        id: z.string().min(1).describe("Vast.ai instance id")
      }
    },
    safeHandler(async ({ id }) => {
      const instance = await showVastInstance(id, options);
      return asToolText(extractVastConnectionInfo(instance));
    })
  );

  server.registerTool(
    "vast_vllm_health",
    {
      description: "Check an OpenAI-compatible vLLM endpoint by calling /v1/models.",
      inputSchema: {
        baseUrl: z.string().url().describe("Endpoint base URL, for example http://1.2.3.4:18000"),
        timeoutMs: z.number().int().min(250).max(30000).optional().describe("Request timeout in milliseconds")
      }
    },
    safeHandler(async ({ baseUrl, timeoutMs }) => asToolText(await checkVllmHealth(baseUrl, { ...options, timeoutMs })))
  );

  return server;
}

export async function startVastMcpServer(options: VastMcpOptions = {}): Promise<{ close: () => Promise<void> }> {
  const server = await createVastMcpServer(options);
  const transport = new StdioServerTransport();
  await server.connect(transport);
  return {
    close: async () => {
      await server.close();
    }
  };
}

export async function getVastStatus(options: VastMcpOptions = {}) {
  const authSource = await getAuthSource(options);
  const cli = await getCliStatus(options);
  return {
    apiBaseUrl: options.apiBaseUrl ?? DEFAULT_API_BASE_URL,
    authSource,
    cli,
    readOnly: true,
    tools: ["vast_list_instances", "vast_show_instance", "vast_connection_info", "vast_vllm_health"]
  };
}

export async function listVastInstances(options: VastMcpOptions = {}) {
  const apiKey = options.apiKey ?? process.env.VAST_API_KEY;
  if (apiKey) {
    const payload = await vastApiRequest("/api/v1/instances/", options);
    return normalizeInstanceList(payload);
  }

  const payload = await runVastCliJson(["show", "instances", "--raw"], options);
  return normalizeInstanceList(payload);
}

export async function showVastInstance(id: string, options: VastMcpOptions = {}) {
  const apiKey = options.apiKey ?? process.env.VAST_API_KEY;
  if (apiKey) {
    const payload = await vastApiRequest(`/api/v0/instances/${encodeURIComponent(id)}/`, options);
    return normalizeInstanceRecord(payload);
  }

  const payload = await runVastCliJson(["show", "instance", id, "--raw"], options);
  return normalizeInstanceRecord(payload);
}

export function extractVastConnectionInfo(instance: unknown): VastConnectionInfo {
  const record = isRecord(instance) ? instance : {};
  const instanceId = readFirst(record, ["id", "instance_id", "machine_id"]);
  const status = readFirstString(record, ["actual_status", "status", "cur_state"]);
  const sshHost = readFirstString(record, ["ssh_host", "public_ipaddr", "public_ip", "host_ip", "hostname"]);
  const sshPort = readFirstNumber(record, ["ssh_port", "ssh_port_external", "direct_port"]);
  const ports = collectPorts(record);
  const apiCandidates = ports
    .filter((port) => port.containerPort === 8000 || port.containerPort === 18000 || port.label?.toLowerCase().includes("vllm"))
    .map((port) => port.url)
    .filter((url): url is string => typeof url === "string" && url.length > 0);

  return {
    apiCandidates: unique(apiCandidates),
    instanceId: typeof instanceId === "string" || typeof instanceId === "number" ? instanceId : undefined,
    ports,
    sshCommand: sshHost && sshPort ? `ssh -p ${sshPort} root@${sshHost}` : undefined,
    sshHost,
    sshPort,
    status
  };
}

export async function checkVllmHealth(baseUrl: string, options: VastMcpOptions & { timeoutMs?: number } = {}) {
  const fetchFn = options.fetch ?? fetch;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? 5000);
  const modelsUrl = new URL("/v1/models", ensureTrailingSlash(baseUrl));

  try {
    const response = await fetchFn(modelsUrl, { signal: controller.signal });
    const text = await response.text();
    return {
      ok: response.ok,
      status: response.status,
      url: modelsUrl.toString(),
      payload: parseJsonOrText(text)
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function vastApiRequest(path: string, options: VastMcpOptions) {
  const apiKey = options.apiKey ?? process.env.VAST_API_KEY;
  if (!apiKey) {
    throw new Error("VAST_API_KEY is not set. Configure it or run `vastai set api-key` for CLI fallback.");
  }

  const fetchFn = options.fetch ?? fetch;
  const url = new URL(path, options.apiBaseUrl ?? DEFAULT_API_BASE_URL);
  const response = await fetchFn(url, {
    headers: {
      Authorization: `Bearer ${apiKey}`,
      Accept: "application/json"
    }
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Vast.ai API ${response.status}: ${text.slice(0, 500)}`);
  }
  return parseJsonOrText(text);
}

async function runVastCliJson(args: string[], options: VastMcpOptions) {
  const stdout = options.runCli ? await options.runCli(args) : await runVastCli(args, options);
  return parseJsonOrText(stdout);
}

async function runVastCli(args: string[], options: VastMcpOptions) {
  const { stdout } = await execFileAsync(options.cliPath ?? DEFAULT_CLI_PATH, args, {
    maxBuffer: 1024 * 1024 * 10,
    timeout: 30000
  });
  return stdout;
}

async function getAuthSource(options: VastMcpOptions) {
  if (options.apiKey) {
    return "options.apiKey";
  }
  if (process.env.VAST_API_KEY) {
    return "VAST_API_KEY";
  }
  try {
    await fs.access(`${os.homedir()}/.config/vastai/vast_api_key`);
    return "~/.config/vastai/vast_api_key";
  } catch {
    return "not_configured";
  }
}

async function getCliStatus(options: VastMcpOptions) {
  try {
    const stdout = options.runCli ? await options.runCli(["--help"]) : await runVastCli(["--help"], options);
    return {
      available: true,
      command: options.cliPath ?? DEFAULT_CLI_PATH,
      helpPreview: stdout.split("\n").slice(0, 3).join("\n")
    };
  } catch (error) {
    return {
      available: false,
      command: options.cliPath ?? DEFAULT_CLI_PATH,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

function normalizeInstanceList(payload: unknown) {
  if (Array.isArray(payload)) {
    return payload;
  }
  if (isRecord(payload)) {
    if (Array.isArray(payload.instances)) {
      return payload.instances;
    }
    if (Array.isArray(payload.results)) {
      return payload.results;
    }
    if (isRecord(payload.instances)) {
      return [payload.instances];
    }
  }
  return payload;
}

function normalizeInstanceRecord(payload: unknown) {
  if (isRecord(payload) && isRecord(payload.instances)) {
    return payload.instances;
  }
  if (isRecord(payload) && isRecord(payload.instance)) {
    return payload.instance;
  }
  return payload;
}

function collectPorts(record: Record<string, unknown>): VastPortMapping[] {
  const mappings: VastPortMapping[] = [];

  for (const key of ["ports", "port_map", "port_mappings", "docker_ports"]) {
    const value = record[key];
    collectPortsFromValue(value, mappings);
  }

  for (const [key, value] of Object.entries(record)) {
    if (/port/i.test(key)) {
      collectPortsFromValue(value, mappings, key);
    }
  }

  return dedupePorts(mappings);
}

function collectPortsFromValue(value: unknown, mappings: VastPortMapping[], label?: string) {
  if (Array.isArray(value)) {
    for (const item of value) {
      collectPortsFromValue(item, mappings, label);
    }
    return;
  }

  if (!isRecord(value)) {
    return;
  }

  for (const [key, child] of Object.entries(value)) {
    const containerPort = parsePort(key) ?? readFirstNumber(value, ["containerPort", "container_port", "private_port"]);
    if (containerPort && (Array.isArray(child) || isRecord(child))) {
      collectPortMapping(containerPort, child, mappings, label ?? key);
      continue;
    }
    if (isRecord(child) || Array.isArray(child)) {
      collectPortsFromValue(child, mappings, label ?? key);
    }
  }

  const containerPort = readFirstNumber(value, ["containerPort", "container_port", "private_port"]);
  if (containerPort) {
    collectPortMapping(containerPort, value, mappings, label);
  }
}

function collectPortMapping(containerPort: number, value: unknown, mappings: VastPortMapping[], label?: string) {
  if (Array.isArray(value)) {
    for (const item of value) {
      collectPortMapping(containerPort, item, mappings, label);
    }
    return;
  }
  if (!isRecord(value)) {
    return;
  }

  const host = readFirstString(value, ["HostIp", "hostIp", "host", "ip", "public_ipaddr"]);
  const hostPort = readFirstNumber(value, ["HostPort", "hostPort", "port", "public_port", "external_port"]);
  mappings.push({
    containerPort,
    host,
    hostPort,
    label,
    url: host && hostPort ? `http://${host}:${hostPort}` : undefined
  });
}

function dedupePorts(ports: VastPortMapping[]) {
  const seen = new Set<string>();
  const deduped: VastPortMapping[] = [];
  for (const port of ports) {
    const key = `${port.containerPort}:${port.host ?? ""}:${port.hostPort ?? ""}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(port);
  }
  return deduped.sort((a, b) => a.containerPort - b.containerPort);
}

function parsePort(value: string) {
  const match = value.match(/^(\d+)(?:\/tcp)?$/i);
  if (!match) {
    return undefined;
  }
  const port = Number(match[1]);
  return Number.isFinite(port) ? port : undefined;
}

function parseJsonOrText(text: string) {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function ensureTrailingSlash(value: string) {
  return value.endsWith("/") ? value : `${value}/`;
}

function readFirst(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (value !== undefined && value !== null && value !== "") {
      return value;
    }
  }
  return undefined;
}

function readFirstString(record: Record<string, unknown>, keys: string[]) {
  const value = readFirst(record, keys);
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number") {
    return String(value);
  }
  return undefined;
}

function readFirstNumber(record: Record<string, unknown>, keys: string[]) {
  const value = readFirst(record, keys);
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function unique(values: string[]) {
  return [...new Set(values)];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function asToolText(value: unknown) {
  return {
    content: [
      {
        type: "text" as const,
        text: JSON.stringify(value, (_key, item) => (item === undefined ? null : item), 2) ?? "null"
      }
    ]
  };
}

function asToolError(message: string) {
  return {
    isError: true,
    content: [
      {
        type: "text" as const,
        text: message
      }
    ]
  };
}

function safeHandler<Args, Result>(
  handler: (args: Args) => Promise<Result>
): (args: Args) => Promise<Result | ReturnType<typeof asToolError>> {
  return async (args: Args) => {
    try {
      return await handler(args);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[vast-mcp] tool handler failed: ${message}`);
      return asToolError(message);
    }
  };
}
