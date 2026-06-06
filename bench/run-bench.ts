import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { checkVastImaginationHealth, type VastImaginationClientOptions } from "../src/index.js";
import {
  type BenchStep,
  type BenchStepResult,
  createSimulatedImagination,
  formatSummaryTable,
  runBenchmark,
  toCsv
} from "./bench.js";

interface CliOptions {
  tracePath: string;
  outPath: string;
  goal: string;
  offline: boolean;
  repeat: number;
  maxBranches: number;
  simDraftMs: number;
  simVerifyMs: number;
  draftUrl?: string;
  verifierUrl?: string;
}

const DEFAULT_OUT = "bench/results.csv";
const DEFAULT_GOAL = "diagnose and fix a failing TypeScript build, then ship it";
const DEFAULT_MAX_BRANCHES = 4;
const DEFAULT_SIM_DRAFT_MS = 40;
const DEFAULT_SIM_VERIFY_MS = 6;
const SAMPLE_TRACE_URL = new URL("sample-trace.json", import.meta.url);

const HELP = `psd-common-sense benchmark runner

Usage: npm run bench -- [options]

  --trace <path>        Trace JSON (array of {action, goal?, state?}). Default: bundled sample.
  --out <path>          CSV output path. Default: ${DEFAULT_OUT}
  --goal <text>         Overall goal applied when a step omits its own.
  --offline             Dry-run with a simulated GPU (no endpoints needed). Latencies are SIMULATED.
  --sim-draft-ms <n>    Simulated draft latency in offline mode. Default: ${DEFAULT_SIM_DRAFT_MS}
  --sim-verify-ms <n>   Simulated verify latency in offline mode. Default: ${DEFAULT_SIM_VERIFY_MS}
  --repeat <n>          Repeat the trace n times to grow the step count. Default: 1
  --max-branches <n>    Branches drafted per step. Default: ${DEFAULT_MAX_BRANCHES}
  --draft-url <url>     Draft vLLM base URL (else VAST_DRAFT_BASE_URL / localhost:18000).
  --verifier-url <url>  Verifier vLLM base URL (else VAST_VERIFIER_BASE_URL / localhost:18001).
  --help                Show this help.

Real run (after opening the SSH tunnel to your Vast box):
  npm run bench -- --trace bench/sample-trace.json --out bench/results.csv
`;

export function parseArgs(argv: string[]): CliOptions {
  const get = (flag: string) => {
    const at = argv.indexOf(flag);
    return at >= 0 ? argv[at + 1] : undefined;
  };
  const has = (flag: string) => argv.includes(flag);

  return {
    tracePath: get("--trace") ?? fileURLToPath(SAMPLE_TRACE_URL),
    outPath: get("--out") ?? DEFAULT_OUT,
    goal: get("--goal") ?? DEFAULT_GOAL,
    offline: has("--offline"),
    repeat: Math.max(1, Number(get("--repeat") ?? 1) || 1),
    maxBranches: Math.max(1, Number(get("--max-branches") ?? DEFAULT_MAX_BRANCHES) || DEFAULT_MAX_BRANCHES),
    simDraftMs: Number(get("--sim-draft-ms") ?? DEFAULT_SIM_DRAFT_MS) || DEFAULT_SIM_DRAFT_MS,
    simVerifyMs: Number(get("--sim-verify-ms") ?? DEFAULT_SIM_VERIFY_MS) || DEFAULT_SIM_VERIFY_MS,
    draftUrl: get("--draft-url"),
    verifierUrl: get("--verifier-url")
  };
}

async function loadTrace(path: string, repeat: number): Promise<BenchStep[]> {
  const raw = await readFile(path, "utf8");
  const parsed: unknown = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error(`Trace at ${path} must be a JSON array of steps.`);
  }
  const steps = parsed.map((entry, index) => normalizeStep(entry, index));
  return Array.from({ length: repeat }, () => steps).flat();
}

function normalizeStep(entry: unknown, index: number): BenchStep {
  if (typeof entry === "string") {
    return { action: entry };
  }
  if (!entry || typeof entry !== "object" || typeof (entry as BenchStep).action !== "string") {
    throw new Error(`Trace step ${index} must be a string or an object with an "action".`);
  }
  return entry as BenchStep;
}

function logStep(result: BenchStepResult): void {
  if (result.error) {
    console.log(`  ${pad(result.index)} ✗ ${truncate(result.action)} — ${result.error}`);
    return;
  }
  const verdicts = `r${result.readyBranches}/b${result.blockedBranches}`;
  const prefetch = result.prefetchAttempts > 0 ? (result.prefetchHits >= result.prefetchAttempts ? "hit" : "part") : "—";
  console.log(
    `  ${pad(result.index)} ${truncate(result.action)} | drafted ${result.draftedBranches} (${verdicts}) | ` +
      `draft ${result.draftLatencyMs}ms | cse ${result.avgCseLatencyMs}ms | prefetch ${prefetch}`
  );
}

async function ensureEndpointsReachable(imagination: VastImaginationClientOptions): Promise<void> {
  const health = await checkVastImaginationHealth({
    draftBaseUrl: imagination.draftBaseUrl,
    verifierBaseUrl: imagination.verifierBaseUrl
  });
  if (!health.ok) {
    throw new Error(
      `vLLM endpoints not reachable (draft=${health.draft.ok}, verifier=${health.verifier.ok}). ` +
        "Start the servers and open the SSH tunnel, or pass --offline for a simulated dry run."
    );
  }
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (argv.includes("--help")) {
    console.log(HELP);
    return;
  }

  const options = parseArgs(argv);
  const trace = await loadTrace(options.tracePath, options.repeat);

  let imagination: VastImaginationClientOptions;
  if (options.offline) {
    imagination = createSimulatedImagination({ draftLatencyMs: options.simDraftMs, verifyLatencyMs: options.simVerifyMs });
    console.log(`Mode: OFFLINE (simulated GPU — latencies are NOT real)\nSteps: ${trace.length}\n`);
  } else {
    imagination = { draftBaseUrl: options.draftUrl, verifierBaseUrl: options.verifierUrl };
    await ensureEndpointsReachable(imagination);
    console.log(`Mode: LIVE GPU\nSteps: ${trace.length}\n`);
  }

  const { results, summary } = await runBenchmark({
    trace,
    goal: options.goal,
    imagination,
    parallelPrefetch: true,
    maxBranches: options.maxBranches,
    onStep: logStep
  });

  await writeFile(options.outPath, toCsv(results), "utf8");
  console.log(`\n${formatSummaryTable(summary)}`);
  console.log(`Per-step CSV: ${options.outPath}`);
  if (options.offline) {
    console.log("\nNote: OFFLINE numbers are simulated. Re-run against a live GPU for publishable latency.");
  }
}

function pad(index: number): string {
  return String(index).padStart(3, " ");
}

function truncate(text: string, max = 42): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text.padEnd(max, " ");
}

const isMain =
  typeof process !== "undefined" &&
  Boolean(process.argv[1]) &&
  (process.argv[1].endsWith("run-bench.ts") || process.argv[1].endsWith("run-bench.js"));

if (isMain) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
