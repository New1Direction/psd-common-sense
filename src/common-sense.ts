export type CommonSenseAction =
  | string
  | {
      command?: string;
      description?: string;
      parameters?: Record<string, unknown>;
      target?: string;
      tool?: string;
    };

export type CommonSenseWorldState = Record<string, unknown>;

export type CommonSenseDecision = "approve" | "warn" | "ask_human" | "block";

export type CommonSenseSeverity = "low" | "medium" | "high" | "critical";

export type CommonSenseHorizon = "immediate" | "near_term" | "downstream";

export interface CommonSenseInput {
  action: CommonSenseAction;
  goal?: string;
  state: CommonSenseWorldState;
}

export interface CommonSenseRisk {
  evidence: string[];
  id: string;
  label: string;
  reason: string;
  reversible: boolean;
  score: number;
  severity: CommonSenseSeverity;
}

export interface CommonSenseOutcome {
  confidence: number;
  evidence: string[];
  horizon: CommonSenseHorizon;
  id: string;
  label: string;
}

export interface CommonSenseAlternative {
  action: string;
  expectedRiskDelta: number;
  rationale: string;
}

export interface CommonSenseFacet {
  confidence: number;
  name: "causal" | "constraint" | "goal" | "reversibility" | "risk" | "social" | "temporal";
  signal: string;
}

export interface CommonSenseTransition {
  action: string;
  evidence: string[];
  expectedOutcomeScore: number;
  predictedStateAfter: string[];
  stateBefore: CommonSenseWorldState;
}

export interface CommonSenseForecast {
  confidence: number;
  decision: CommonSenseDecision;
  facets: CommonSenseFacet[];
  impossible: string[];
  likely: string[];
  plausible: string[];
  reversibility: number;
  risk: number;
  risks: CommonSenseRisk[];
  saferAlternatives: CommonSenseAlternative[];
  successProbability: number;
  transition: CommonSenseTransition;
  whyThisMayBeBad: string[];
}

interface Facts {
  backupFresh?: boolean;
  branch?: string;
  currentBackup?: boolean;
  environment?: string;
  packageAlternatives?: number;
  packageDeprecated?: boolean;
  packageLastUpdatedYearsAgo?: number;
  production: boolean;
  repoDirty?: boolean;
  reviewed?: boolean;
  testsPassing?: boolean;
}

interface MutableForecast {
  confidenceSignals: number;
  facets: CommonSenseFacet[];
  impossible: CommonSenseOutcome[];
  likely: CommonSenseOutcome[];
  plausible: CommonSenseOutcome[];
  reversibilityDelta: number;
  riskDelta: number;
  risks: CommonSenseRisk[];
  saferAlternatives: CommonSenseAlternative[];
  successDelta: number;
}

const DEFAULT_RISK = 0.08;
const DEFAULT_REVERSIBILITY = 0.72;
const DEFAULT_SUCCESS_PROBABILITY = 0.64;

export function forecastCommonSense(input: CommonSenseInput): CommonSenseForecast {
  const action = normalizeAction(input.action);
  const facts = readFacts(input.state);
  const forecast: MutableForecast = {
    confidenceSignals: 0,
    facets: [],
    impossible: [],
    likely: [],
    plausible: [],
    reversibilityDelta: 0,
    riskDelta: 0,
    risks: [],
    saferAlternatives: [],
    successDelta: 0
  };

  addGeneralPlausibility(input, action, facts, forecast);
  addGitSense(action, facts, forecast);
  addDeploymentSense(action, facts, forecast);
  addDestructiveActionSense(action, facts, forecast);
  addPackageSense(action, facts, forecast);
  addShellTrustSense(action, forecast);
  addSocialSense(action, facts, forecast);
  addGoalSense(input.goal, action, facts, forecast);

  if (forecast.likely.length === 0) {
    addOutcome(forecast, "likely", {
      confidence: 0.56,
      evidence: ["No high-risk action pattern matched."],
      horizon: "immediate",
      id: "action_can_execute",
      label: "The action can probably execute as requested."
    });
  }

  if (forecast.plausible.length === 0) {
    addOutcome(forecast, "plausible", {
      confidence: 0.52,
      evidence: ["Limited world-state evidence was provided."],
      horizon: "near_term",
      id: "unknown_side_effects",
      label: "Unobserved side effects may still exist outside the supplied state."
    });
  }

  const risk = clamp01(DEFAULT_RISK + forecast.riskDelta);
  const reversibility = clamp01(DEFAULT_REVERSIBILITY + forecast.reversibilityDelta - risk * 0.28);
  const successProbability = clamp01(DEFAULT_SUCCESS_PROBABILITY + forecast.successDelta - risk * 0.34);
  const confidence = clamp01(0.48 + Math.min(0.42, forecast.confidenceSignals * 0.045));
  const decision = decide(risk, reversibility, forecast.risks);
  const predictedStateAfter = [
    ...forecast.likely.map((outcome) => outcome.label),
    ...forecast.plausible.slice(0, 3).map((outcome) => outcome.label)
  ];

  return {
    confidence: round2(confidence),
    decision,
    facets: compactFacets(forecast.facets),
    impossible: uniqueLabels(forecast.impossible),
    likely: uniqueLabels(forecast.likely),
    plausible: uniqueLabels(forecast.plausible),
    reversibility: round2(reversibility),
    risk: round2(risk),
    risks: mergeRisks(forecast.risks),
    saferAlternatives: mergeAlternatives(forecast.saferAlternatives),
    successProbability: round2(successProbability),
    transition: {
      action,
      evidence: buildTransitionEvidence(facts, action),
      expectedOutcomeScore: round2(successProbability - risk),
      predictedStateAfter,
      stateBefore: input.state
    },
    whyThisMayBeBad: buildBadIdeaSummary(forecast.risks, forecast.impossible)
  };
}

export function recordCommonSenseTransition(
  input: CommonSenseInput,
  forecast: CommonSenseForecast,
  observed?: { outcomeScore?: number; stateAfter?: CommonSenseWorldState }
) {
  return {
    action: forecast.transition.action,
    expectedOutcomeScore: forecast.transition.expectedOutcomeScore,
    observedOutcomeScore: observed?.outcomeScore,
    observedStateAfter: observed?.stateAfter,
    predictedStateAfter: forecast.transition.predictedStateAfter,
    stateAfter: observed?.stateAfter,
    stateBefore: input.state
  };
}

export function summarizeCommonSenseForecast(forecast: CommonSenseForecast): string {
  const lines = [
    `Decision: ${forecast.decision}`,
    `Risk: ${forecast.risk.toFixed(2)} | Reversibility: ${forecast.reversibility.toFixed(2)} | Success: ${forecast.successProbability.toFixed(2)}`,
    "Likely:",
    ...forecast.likely.map((item) => `- ${item}`),
    "Risks:",
    ...forecast.risks.map((risk) => `- [${risk.severity}] ${risk.label}: ${risk.reason}`)
  ];

  if (forecast.saferAlternatives.length > 0) {
    lines.push("Safer alternatives:", ...forecast.saferAlternatives.map((alt) => `- ${alt.action}: ${alt.rationale}`));
  }

  return `${lines.join("\n")}\n`;
}

function addGeneralPlausibility(input: CommonSenseInput, action: string, facts: Facts, forecast: MutableForecast) {
  if (action.length > 0) {
    addFacet(forecast, "causal", "A concrete action was supplied, so direct consequences can be forecast.", 0.62);
  }

  if (Object.keys(input.state).length > 0) {
    addFacet(forecast, "constraint", "The supplied world state gives constraints to test the action against.", 0.64);
  }

  if (facts.production) {
    addFacet(forecast, "risk", "The action touches a production-like environment.", 0.78);
  }
}

function addGitSense(action: string, facts: Facts, forecast: MutableForecast) {
  const branch = facts.branch?.toLowerCase();
  const onSharedBranch = branch === "main" || branch === "master" || branch === "trunk" || branch === "production";
  const forcePush =
    hasAny(action, ["git push --force", "git push -f", "force push"]) || (action.includes("git push") && action.includes("--force"));
  const forceWithLease = action.includes("--force-with-lease");

  if (forcePush && !forceWithLease) {
    addOutcome(forecast, "likely", {
      confidence: 0.86,
      evidence: ["Action includes a force-push pattern."],
      horizon: "immediate",
      id: "force_push_overwrites_history",
      label: "Force push can overwrite remote history."
    });
    addRisk(forecast, {
      evidence: ["Action includes --force or an equivalent force-push phrase."],
      id: "git_history_loss",
      label: "Remote history loss",
      reason: "Collaborators or automation may lose commits if the remote branch is rewritten.",
      reversible: false,
      score: 0.34,
      severity: "high"
    });
    addAlternative(forecast, {
      action: "Create a backup branch, then use git push --force-with-lease only after checking the remote.",
      expectedRiskDelta: -0.28,
      rationale: "It preserves a recovery point and refuses to overwrite unseen remote work."
    });
    forecast.reversibilityDelta -= 0.24;
    forecast.successDelta -= 0.08;
    addFacet(forecast, "reversibility", "History rewrites are reversible only if a usable reference still exists.", 0.8);
  }

  if (forcePush && onSharedBranch) {
    addOutcome(forecast, "impossible", {
      confidence: 0.82,
      evidence: [`Current branch is ${facts.branch}.`, "Action rewrites remote history."],
      horizon: "immediate",
      id: "risk_free_shared_force_push",
      label: "A risk-free force push to a shared mainline branch."
    });
    addRisk(forecast, {
      evidence: [`Current branch is ${facts.branch}.`],
      id: "shared_branch_rewrite",
      label: "Shared branch rewrite",
      reason: "Mainline branches are usually consumed by teammates, CI, release systems, or deployment automation.",
      reversible: false,
      score: 0.26,
      severity: "critical"
    });
    forecast.reversibilityDelta -= 0.18;
  }

  if (facts.repoDirty && hasAny(action, ["git push", "git reset", "git checkout", "git clean", "deploy", "release"])) {
    addOutcome(forecast, "plausible", {
      confidence: 0.72,
      evidence: ["repo_dirty is true."],
      horizon: "near_term",
      id: "dirty_tree_confuses_recovery",
      label: "Uncommitted local changes may make rollback or diagnosis ambiguous."
    });
    addRisk(forecast, {
      evidence: ["The repository has uncommitted changes."],
      id: "dirty_worktree",
      label: "Dirty working tree",
      reason: "The current filesystem state may not match the committed state being shared or deployed.",
      reversible: true,
      score: 0.12,
      severity: "medium"
    });
    addAlternative(forecast, {
      action: "Commit, stash, or explicitly discard local changes before the operation.",
      expectedRiskDelta: -0.12,
      rationale: "A clean tree makes cause and rollback much easier to reason about."
    });
  }
}

function addDeploymentSense(action: string, facts: Facts, forecast: MutableForecast) {
  const deployLike = hasAny(action, ["deploy", "release", "rollout", "publish", "git push"]);

  if (deployLike && facts.testsPassing === false) {
    addOutcome(forecast, "likely", {
      confidence: 0.84,
      evidence: ["tests_passing is false.", "Action can publish or propagate changes."],
      horizon: "near_term",
      id: "known_failing_change_propagates",
      label: "The action may propagate code that is already known to fail tests."
    });
    addRisk(forecast, {
      evidence: ["Tests are not passing."],
      id: "known_test_failure",
      label: "Known failing tests",
      reason: "Failing tests mean at least one expected behavior is already broken or unverified.",
      reversible: true,
      score: 0.24,
      severity: facts.production ? "critical" : "high"
    });
    addAlternative(forecast, {
      action: "Run and fix the failing test suite before publishing the change.",
      expectedRiskDelta: -0.22,
      rationale: "It converts a known-bad state into evidence that the goal is safe to continue."
    });
    forecast.successDelta -= 0.22;
    addFacet(forecast, "goal", "Deploying while tests fail conflicts with the goal of shipping a working feature.", 0.78);
  }

  if (deployLike && facts.production) {
    addRisk(forecast, {
      evidence: [`Environment is ${facts.environment ?? "production-like"}.`],
      id: "production_change",
      label: "Production blast radius",
      reason: "Mistakes can affect real users, data, or revenue.",
      reversible: true,
      score: 0.16,
      severity: "high"
    });
    addFacet(forecast, "temporal", "Production failures have immediate user-visible effects.", 0.72);
  }
}

function addDestructiveActionSense(action: string, facts: Facts, forecast: MutableForecast) {
  const destructive = hasAny(action, [
    "delete",
    "destroy",
    "drop table",
    "drop database",
    "rm -rf",
    "truncate",
    "wipe",
    "purge",
    "overwrite"
  ]);

  if (!destructive) {
    return;
  }

  addOutcome(forecast, "likely", {
    confidence: 0.82,
    evidence: ["Action uses destructive language or a destructive command."],
    horizon: "immediate",
    id: "state_removed_or_overwritten",
    label: "The action removes or overwrites existing state."
  });
  addRisk(forecast, {
    evidence: ["The action is destructive."],
    id: "destructive_action",
    label: "Irreversible state change",
    reason: "Deleted or overwritten state may not be recoverable from the current system alone.",
    reversible: false,
    score: 0.3,
    severity: "high"
  });
  forecast.reversibilityDelta -= 0.28;
  forecast.successDelta -= 0.08;
  addFacet(forecast, "causal", "Destructive actions directly reduce the available future states.", 0.74);

  if (facts.currentBackup === false || facts.backupFresh === false || facts.backupFresh === undefined) {
    addOutcome(forecast, "impossible", {
      confidence: facts.backupFresh === false || facts.currentBackup === false ? 0.9 : 0.64,
      evidence: buildBackupEvidence(facts),
      horizon: "downstream",
      id: "guaranteed_safe_rollback_without_backup",
      label: "Guaranteed safe rollback without a current verified backup."
    });
    addRisk(forecast, {
      evidence: buildBackupEvidence(facts),
      id: "missing_verified_backup",
      label: "Missing verified backup",
      reason: "Recovery depends on backup freshness and restore validity, not just intent.",
      reversible: false,
      score: facts.production ? 0.34 : 0.22,
      severity: facts.production ? "critical" : "high"
    });
    addAlternative(forecast, {
      action: "Take and verify a backup, then perform the destructive change inside a reversible migration plan.",
      expectedRiskDelta: -0.32,
      rationale: "The backup creates an actual recovery path instead of an assumed one."
    });
  }

  if (facts.production) {
    addOutcome(forecast, "plausible", {
      confidence: 0.78,
      evidence: [`Environment is ${facts.environment ?? "production-like"}.`, "Action is destructive."],
      horizon: "near_term",
      id: "production_outage_or_data_loss",
      label: "A production outage or data-loss incident could follow."
    });
    forecast.successDelta -= 0.18;
  }
}

function addPackageSense(action: string, facts: Facts, forecast: MutableForecast) {
  if (!hasAny(action, ["npm install", "pnpm add", "yarn add", "bun add"])) {
    return;
  }

  addOutcome(forecast, "plausible", {
    confidence: 0.58,
    evidence: ["Action installs a new dependency."],
    horizon: "downstream",
    id: "dependency_surface_area_increases",
    label: "The dependency surface area and supply-chain exposure increase."
  });
  addRisk(forecast, {
    evidence: ["New package installation changes the trusted codebase."],
    id: "new_dependency",
    label: "New dependency trust",
    reason: "Installed packages execute code and can bring vulnerabilities, maintenance drag, or transitive risk.",
    reversible: true,
    score: 0.12,
    severity: "medium"
  });

  if (facts.packageDeprecated) {
    addRisk(forecast, {
      evidence: ["package_deprecated is true."],
      id: "deprecated_dependency",
      label: "Deprecated dependency",
      reason: "Deprecated packages usually indicate known maintenance or replacement concerns.",
      reversible: true,
      score: 0.18,
      severity: "high"
    });
  }

  if (facts.packageLastUpdatedYearsAgo !== undefined && facts.packageLastUpdatedYearsAgo >= 3) {
    addRisk(forecast, {
      evidence: [`Package last update was ${facts.packageLastUpdatedYearsAgo} years ago.`],
      id: "stale_dependency",
      label: "Stale dependency",
      reason: "Long-stale packages are less likely to track platform, security, and ecosystem changes.",
      reversible: true,
      score: 0.16,
      severity: "high"
    });
  }

  if (facts.packageAlternatives !== undefined && facts.packageAlternatives > 0) {
    addAlternative(forecast, {
      action: "Compare maintained alternatives before installing this package.",
      expectedRiskDelta: -0.14,
      rationale: `${facts.packageAlternatives} known alternative${facts.packageAlternatives === 1 ? "" : "s"} may offer a better maintenance profile.`
    });
  }

  addFacet(forecast, "risk", "Dependency choices affect future security and maintenance, not just the current task.", 0.7);
}

function addShellTrustSense(action: string, forecast: MutableForecast) {
  if (!hasAny(action, ["curl"]) || !hasAny(action, ["| sh", "| bash", "bash -c", "sh -c"])) {
    return;
  }

  addOutcome(forecast, "likely", {
    confidence: 0.78,
    evidence: ["Action pipes downloaded content into a shell."],
    horizon: "immediate",
    id: "remote_code_executes",
    label: "Remote code executes locally before it can be fully inspected."
  });
  addRisk(forecast, {
    evidence: ["The command combines network fetch and shell execution."],
    id: "unreviewed_remote_code",
    label: "Unreviewed remote code execution",
    reason: "The executed script can change files, install software, or exfiltrate data using current permissions.",
    reversible: false,
    score: 0.3,
    severity: "critical"
  });
  addAlternative(forecast, {
    action: "Download the script, inspect it, pin its checksum, then run it with the minimum required permissions.",
    expectedRiskDelta: -0.26,
    rationale: "Separating fetch, review, and execution restores human auditability."
  });
  forecast.reversibilityDelta -= 0.18;
}

function addSocialSense(action: string, facts: Facts, forecast: MutableForecast) {
  if (!hasAny(action, ["send email", "email", "message customer", "publish post", "announce"])) {
    return;
  }

  addFacet(forecast, "social", "Human-facing actions can create reputational or coordination consequences.", 0.68);

  if (facts.reviewed === false) {
    addRisk(forecast, {
      evidence: ["reviewed is false."],
      id: "unreviewed_human_facing_action",
      label: "Unreviewed human-facing communication",
      reason: "A mistaken public or customer-facing message can be hard to fully retract.",
      reversible: false,
      score: 0.2,
      severity: "high"
    });
    addAlternative(forecast, {
      action: "Draft the message and request human review before sending.",
      expectedRiskDelta: -0.18,
      rationale: "Review catches tone, accuracy, privacy, and audience mistakes before they leave the system."
    });
  }
}

function addGoalSense(goal: string | undefined, action: string, facts: Facts, forecast: MutableForecast) {
  if (!goal) {
    return;
  }

  const normalizedGoal = goal.toLowerCase();
  const deployGoal = hasAny(normalizedGoal, ["deploy", "ship", "release"]);
  const destructiveAction = hasAny(action, ["drop table", "delete", "rm -rf", "truncate", "wipe"]);

  if (deployGoal && facts.testsPassing === false) {
    addOutcome(forecast, "plausible", {
      confidence: 0.74,
      evidence: [`Goal is "${goal}".`, "tests_passing is false."],
      horizon: "near_term",
      id: "goal_success_unlikely_with_failing_tests",
      label: "The stated goal may be undermined by the known failing test state."
    });
  }

  if (deployGoal && destructiveAction) {
    addRisk(forecast, {
      evidence: [`Goal is "${goal}".`, "Action is destructive."],
      id: "disproportionate_to_goal",
      label: "Disproportionate action",
      reason: "A destructive operation is usually not the smallest safe step toward a deployment goal.",
      reversible: false,
      score: 0.18,
      severity: "high"
    });
    addFacet(forecast, "goal", "The proposed action is more destructive than the stated goal appears to require.", 0.72);
  }
}

function normalizeAction(action: CommonSenseAction): string {
  if (typeof action === "string") {
    return action.trim().replace(/\s+/g, " ");
  }

  return [action.tool, action.command, action.description, action.target, JSON.stringify(action.parameters ?? {})]
    .filter((part): part is string => typeof part === "string" && part.length > 0)
    .join(" ")
    .trim()
    .replace(/\s+/g, " ");
}

function readFacts(state: CommonSenseWorldState): Facts {
  const flat = flattenState(state);
  const environment = readString(flat, ["environment", "env", "targetEnv", "target"]);
  const production = environment
    ? hasAny(environment.toLowerCase(), ["prod", "production"])
    : readBoolean(flat, ["production", "isProduction"]) === true;

  return {
    backupFresh: readBoolean(flat, ["backupFresh", "backupCurrent", "backupsFresh", "backupVerified"]),
    branch: readString(flat, ["branch", "currentBranch", "gitBranch"]),
    currentBackup: readBoolean(flat, ["currentBackup", "verifiedBackup", "hasBackup"]),
    environment,
    packageAlternatives: readNumber(flat, ["packageAlternatives", "knownAlternatives", "popularAlternatives"]),
    packageDeprecated: readBoolean(flat, ["packageDeprecated", "deprecated"]),
    packageLastUpdatedYearsAgo: readNumber(flat, ["packageLastUpdatedYearsAgo", "packageLastUpdateYears", "lastUpdateYearsAgo"]),
    production,
    repoDirty: readDirtyBoolean(flat, ["repoDirty", "workingTreeDirty", "dirty"]),
    reviewed: readBoolean(flat, ["reviewed", "humanReviewed", "approved"]),
    testsPassing: readBoolean(flat, ["testsPassing", "testPassing", "testsGreen", "ciPassing"])
  };
}

function flattenState(state: CommonSenseWorldState) {
  const flat = new Map<string, unknown>();

  function visit(value: unknown, path: string[]) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
        visit(child, [...path, key]);
      }
      return;
    }

    const lastKey = path[path.length - 1];
    if (lastKey) {
      flat.set(normalizeKey(lastKey), value);
    }
    if (path.length > 1) {
      flat.set(normalizeKey(path.join(".")), value);
    }
  }

  visit(state, []);
  return flat;
}

function readBoolean(flat: Map<string, unknown>, aliases: string[]) {
  const value = readValue(flat, aliases);
  return coerceBoolean(value);
}

function readDirtyBoolean(flat: Map<string, unknown>, aliases: string[]) {
  const value = readValue(flat, aliases);
  if (typeof value === "string") {
    const normalized = value.toLowerCase().trim();
    if (normalized === "dirty") {
      return true;
    }
    if (normalized === "clean") {
      return false;
    }
  }
  return coerceBoolean(value);
}

function coerceBoolean(value: unknown) {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalized = value.toLowerCase().trim();
    if (["1", "true", "yes", "y", "passing", "pass", "green", "current", "verified"].includes(normalized)) {
      return true;
    }
    if (["0", "false", "no", "n", "failing", "fail", "red", "stale", "unverified"].includes(normalized)) {
      return false;
    }
  }
  return undefined;
}

function readNumber(flat: Map<string, unknown>, aliases: string[]) {
  const value = readValue(flat, aliases);
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function readString(flat: Map<string, unknown>, aliases: string[]) {
  const value = readValue(flat, aliases);
  return typeof value === "string" ? value : undefined;
}

function readValue(flat: Map<string, unknown>, aliases: string[]) {
  for (const alias of aliases) {
    const value = flat.get(normalizeKey(alias));
    if (value !== undefined) {
      return value;
    }
  }
  return undefined;
}

function normalizeKey(key: string) {
  return key.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function hasAny(value: string, needles: string[]) {
  const normalized = value.toLowerCase();
  return needles.some((needle) => normalized.includes(needle));
}

function addOutcome(forecast: MutableForecast, bucket: "impossible" | "likely" | "plausible", outcome: CommonSenseOutcome) {
  forecast[bucket].push(outcome);
  forecast.confidenceSignals += 1;
}

function addRisk(forecast: MutableForecast, risk: CommonSenseRisk) {
  forecast.risks.push(risk);
  forecast.riskDelta += risk.score;
  forecast.confidenceSignals += risk.severity === "critical" ? 2 : 1;
}

function addAlternative(forecast: MutableForecast, alternative: CommonSenseAlternative) {
  forecast.saferAlternatives.push(alternative);
}

function addFacet(forecast: MutableForecast, name: CommonSenseFacet["name"], signal: string, confidence: number) {
  forecast.facets.push({ confidence, name, signal });
  forecast.confidenceSignals += 0.5;
}

function buildBackupEvidence(facts: Facts) {
  const evidence: string[] = [];
  if (facts.backupFresh === false) {
    evidence.push("backup_fresh is false.");
  }
  if (facts.currentBackup === false) {
    evidence.push("current_backup is false.");
  }
  if (evidence.length === 0) {
    evidence.push("No current verified backup was provided in world state.");
  }
  return evidence;
}

function decide(risk: number, reversibility: number, risks: CommonSenseRisk[]): CommonSenseDecision {
  if (risks.some((risk) => risk.severity === "critical" && risk.score >= 0.26) || (risk >= 0.78 && reversibility < 0.42)) {
    return "block";
  }
  if (risk >= 0.56 || reversibility < 0.3) {
    return "ask_human";
  }
  if (risk >= 0.26) {
    return "warn";
  }
  return "approve";
}

function compactFacets(facets: CommonSenseFacet[]) {
  const byName = new Map<CommonSenseFacet["name"], CommonSenseFacet>();
  for (const facet of facets) {
    const current = byName.get(facet.name);
    if (!current || facet.confidence > current.confidence) {
      byName.set(facet.name, { ...facet, confidence: round2(facet.confidence) });
    }
  }
  return [...byName.values()];
}

function mergeRisks(risks: CommonSenseRisk[]) {
  const merged = new Map<string, CommonSenseRisk>();
  for (const risk of risks) {
    const current = merged.get(risk.id);
    if (!current) {
      merged.set(risk.id, { ...risk, score: round2(risk.score) });
      continue;
    }
    merged.set(risk.id, {
      ...current,
      evidence: unique([...current.evidence, ...risk.evidence]),
      score: round2(Math.max(current.score, risk.score))
    });
  }
  return [...merged.values()].sort((a, b) => b.score - a.score);
}

function mergeAlternatives(alternatives: CommonSenseAlternative[]) {
  const merged = new Map<string, CommonSenseAlternative>();
  for (const alternative of alternatives) {
    const current = merged.get(alternative.action);
    if (!current || alternative.expectedRiskDelta < current.expectedRiskDelta) {
      merged.set(alternative.action, {
        ...alternative,
        expectedRiskDelta: round2(alternative.expectedRiskDelta)
      });
    }
  }
  return [...merged.values()].sort((a, b) => a.expectedRiskDelta - b.expectedRiskDelta);
}

function buildTransitionEvidence(facts: Facts, action: string) {
  const evidence = [`action=${action}`];
  if (facts.branch) {
    evidence.push(`branch=${facts.branch}`);
  }
  if (facts.environment) {
    evidence.push(`environment=${facts.environment}`);
  }
  if (facts.testsPassing !== undefined) {
    evidence.push(`tests_passing=${facts.testsPassing}`);
  }
  if (facts.repoDirty !== undefined) {
    evidence.push(`repo_dirty=${facts.repoDirty}`);
  }
  return evidence;
}

function buildBadIdeaSummary(risks: CommonSenseRisk[], impossible: CommonSenseOutcome[]) {
  return unique([
    ...mergeRisks(risks)
      .filter((risk) => risk.severity === "critical" || risk.severity === "high")
      .map((risk) => risk.reason),
    ...impossible.map((outcome) => outcome.label)
  ]);
}

function uniqueLabels(outcomes: CommonSenseOutcome[]) {
  return unique(outcomes.map((outcome) => outcome.label));
}

function unique(values: string[]) {
  return [...new Set(values)];
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function round2(value: number) {
  return Math.round(value * 100) / 100;
}
