import { checkVllmHealth } from "./vast-mcp.js";

const DEFAULT_DRAFT_BASE_URL = "http://127.0.0.1:18000/v1";
const DEFAULT_VERIFIER_BASE_URL = "http://127.0.0.1:18001/v1";
const DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct";

type FetchLike = (input: string | URL, init?: RequestInit) => Promise<Response>;

export interface VastImaginationBranch {
  action: string;
  label?: string;
  probability?: number;
  rationale?: string;
}

export interface VastImaginationClientOptions {
  draftBaseUrl?: string;
  verifierBaseUrl?: string;
  model?: string;
  apiKey?: string;
  fetch?: FetchLike;
  timeoutMs?: number;
}

export interface VastDraftBranchesInput {
  goal: string;
  state?: Record<string, unknown>;
  currentAction?: string;
  maxBranches?: number;
  systemPrompt?: string;
}

export interface VastDraftBranchesResult {
  branches: VastImaginationBranch[];
  latencyMs: number;
  model: string;
  raw?: unknown;
}

export interface VastVerifyBranchInput {
  action: string;
  goal?: string;
  state?: Record<string, unknown>;
}

export interface VastVerifyBranchResult {
  action: string;
  latencyMs: number;
  model: string;
  notes: string;
  raw?: unknown;
}

export interface VastImaginationHealth {
  draft: Awaited<ReturnType<typeof checkVllmHealth>>;
  ok: boolean;
  verifier: Awaited<ReturnType<typeof checkVllmHealth>>;
}

export function createVastImaginationClient(options: VastImaginationClientOptions = {}) {
  const draftBaseUrl = normalizeV1Base(options.draftBaseUrl ?? process.env.VAST_DRAFT_BASE_URL ?? DEFAULT_DRAFT_BASE_URL);
  const verifierBaseUrl = normalizeV1Base(
    options.verifierBaseUrl ?? process.env.VAST_VERIFIER_BASE_URL ?? DEFAULT_VERIFIER_BASE_URL
  );
  const model = options.model ?? process.env.VAST_PSD_MODEL ?? DEFAULT_MODEL;
  const fetchFn = options.fetch ?? fetch;
  const timeoutMs = options.timeoutMs ?? 120_000;
  const apiKey = options.apiKey ?? process.env.VAST_PSD_API_KEY ?? process.env.OPENAI_API_KEY;

  return {
    draftBaseUrl,
    verifierBaseUrl,
    model,
    checkHealth: () => checkVastImaginationHealth({ draftBaseUrl, verifierBaseUrl, fetch: fetchFn, timeoutMs }),
    draftBranches: (input: VastDraftBranchesInput) =>
      draftBranches(input, { draftBaseUrl, model, fetch: fetchFn, timeoutMs, apiKey }),
    verifyBranch: (input: VastVerifyBranchInput) =>
      verifyBranch(input, { verifierBaseUrl, model, fetch: fetchFn, timeoutMs, apiKey })
  };
}

export async function checkVastImaginationHealth(
  options: Pick<VastImaginationClientOptions, "draftBaseUrl" | "verifierBaseUrl" | "fetch" | "timeoutMs"> = {}
): Promise<VastImaginationHealth> {
  const draftBaseUrl = normalizeV1Base(options.draftBaseUrl ?? process.env.VAST_DRAFT_BASE_URL ?? DEFAULT_DRAFT_BASE_URL);
  const verifierBaseUrl = normalizeV1Base(
    options.verifierBaseUrl ?? process.env.VAST_VERIFIER_BASE_URL ?? DEFAULT_VERIFIER_BASE_URL
  );
  const fetchFn = options.fetch ?? fetch;
  const timeoutMs = options.timeoutMs ?? 5000;
  const [draft, verifier] = await Promise.all([
    checkVllmHealth(stripV1Suffix(draftBaseUrl), { fetch: fetchFn, timeoutMs }),
    checkVllmHealth(stripV1Suffix(verifierBaseUrl), { fetch: fetchFn, timeoutMs })
  ]);
  return {
    draft,
    verifier,
    ok: draft.ok && verifier.ok
  };
}

export async function draftBranches(
  input: VastDraftBranchesInput,
  options: {
    draftBaseUrl: string;
    model: string;
    fetch: FetchLike;
    timeoutMs: number;
    apiKey?: string;
  }
): Promise<VastDraftBranchesResult> {
  const started = Date.now();
  const maxBranches = Math.max(1, Math.min(input.maxBranches ?? 4, 8));
  const prompt = buildDraftPrompt(input, maxBranches);
  const payload = await chatCompletion(options.draftBaseUrl, {
    model: options.model,
    temperature: 0.35,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content:
          input.systemPrompt ??
          "You propose concrete next agent actions. Return JSON only with shape {\"branches\":[{\"action\":\"...\",\"label\":\"...\",\"probability\":0.0-1.0,\"rationale\":\"...\"}]}."
      },
      { role: "user", content: prompt }
    ]
  }, options);

  const branches = parseDraftBranches(payload);
  return {
    branches: branches.slice(0, maxBranches),
    latencyMs: Date.now() - started,
    model: options.model,
    raw: payload
  };
}

export async function verifyBranch(
  input: VastVerifyBranchInput,
  options: {
    verifierBaseUrl: string;
    model: string;
    fetch: FetchLike;
    timeoutMs: number;
    apiKey?: string;
  }
): Promise<VastVerifyBranchResult> {
  const started = Date.now();
  const payload = await chatCompletion(options.verifierBaseUrl, {
    model: options.model,
    temperature: 0.1,
    messages: [
      {
        role: "system",
        content:
          "You are a lightweight verifier for an agent action. Summarize failure modes and whether the action looks executable. Reply in under 80 words."
      },
      {
        role: "user",
        content: JSON.stringify({
          action: input.action,
          goal: input.goal ?? "",
          state: input.state ?? {}
        })
      }
    ]
  }, options);

  return {
    action: input.action,
    latencyMs: Date.now() - started,
    model: options.model,
    notes: extractAssistantText(payload),
    raw: payload
  };
}

async function chatCompletion(
  baseUrl: string,
  body: Record<string, unknown>,
  options: { fetch: FetchLike; timeoutMs: number; apiKey?: string }
) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs);
  const url = new URL("chat/completions", ensureTrailingSlash(baseUrl));

  try {
    const headers: Record<string, string> = {
      Accept: "application/json",
      "Content-Type": "application/json"
    };
    if (options.apiKey) {
      headers.Authorization = `Bearer ${options.apiKey}`;
    }

    const response = await options.fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal
    });
    const text = await response.text();
    if (!response.ok) {
      throw new Error(`vLLM ${response.status} at ${url}: ${text.slice(0, 500)}`);
    }
    return parseJsonOrText(text);
  } finally {
    clearTimeout(timeout);
  }
}

function buildDraftPrompt(input: VastDraftBranchesInput, maxBranches: number) {
  return JSON.stringify({
    goal: input.goal,
    current_action: input.currentAction ?? "",
    world_state: input.state ?? {},
    max_branches: maxBranches,
    instructions: "Propose diverse next actions an autonomous coding agent could take. Prefer inspect/fix/test paths over destructive git or production operations."
  });
}

function parseDraftBranches(payload: unknown): VastImaginationBranch[] {
  const text = extractAssistantText(payload);
  const parsed = tryParseJson(text);
  if (!parsed) {
    return fallbackBranchesFromText(text);
  }

  const record = isRecord(parsed) ? parsed : {};
  const list = Array.isArray(record.branches)
    ? record.branches
    : Array.isArray(record.candidates)
      ? record.candidates
      : Array.isArray(parsed)
        ? parsed
        : [];

  const branches: VastImaginationBranch[] = [];
  for (const item of list) {
    if (typeof item === "string" && item.trim()) {
      branches.push({ action: item.trim() });
      continue;
    }
    if (!isRecord(item)) {
      continue;
    }
    const action = readString(item, ["action", "command", "description"]);
    if (!action) {
      continue;
    }
    branches.push({
      action,
      label: readString(item, ["label", "name"]),
      probability: readNumber(item, ["probability", "score"]),
      rationale: readString(item, ["rationale", "reason", "why"])
    });
  }
  return branches;
}

function fallbackBranchesFromText(text: string): VastImaginationBranch[] {
  const lines = text
    .split("\n")
    .map((line) => line.replace(/^[-*\d.]+\s*/, "").trim())
    .filter((line) => line.length > 3);
  return lines.slice(0, 6).map((action) => ({ action }));
}

function extractAssistantText(payload: unknown) {
  if (!isRecord(payload)) {
    return typeof payload === "string" ? payload : "";
  }
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return "";
  }
  const first = choices[0];
  if (!isRecord(first)) {
    return "";
  }
  const message = first.message;
  if (isRecord(message) && typeof message.content === "string") {
    return message.content;
  }
  if (typeof first.text === "string") {
    return first.text;
  }
  return "";
}

function tryParseJson(text: string) {
  const trimmed = text.trim();
  if (!trimmed) {
    return undefined;
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced?.[1]) {
      try {
        return JSON.parse(fenced[1].trim());
      } catch {
        return undefined;
      }
    }
    return undefined;
  }
}

function normalizeV1Base(url: string) {
  const trimmed = url.replace(/\/+$/, "");
  return trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`;
}

function stripV1Suffix(url: string) {
  return url.replace(/\/v1\/?$/, "");
}

function ensureTrailingSlash(value: string) {
  return value.endsWith("/") ? value : `${value}/`;
}

function parseJsonOrText(text: string) {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function readString(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return undefined;
}

function readNumber(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}