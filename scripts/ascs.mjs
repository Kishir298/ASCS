#!/usr/bin/env node
/**
 * One-command A.S.C.S. launcher: `npm run ASCS`.
 *
 *  1. Ensures `.venv` exists (creates with `py -3.12`, falls back to
 *     `py -3`) and `requirements.txt` is installed; reinstalls only when
 *     the venv python or the `agent` import is broken.
 *  2. Ensures Ollama is up: connects when already running (never owned),
 *     otherwise starts a detached `ollama serve` it owns, polling
 *     `/api/tags` up to 60s. Then verifies `qwen3-coder:30b` (or the
 *     `qwen2.5-coder:14b` fallback) is pulled — fails fast with the exact
 *     `ollama pull` command, never silently pulls GBs.
 *  3. Launches `.venv\Scripts\python.exe -m agent.terminal <args>` with
 *     inherited stdio (required for the curses TUI), forwarding the exit
 *     code. No args → TUI via the terminal entry's `normalize_argv`
 *     default. (NOTE: `python -m agent` bypasses the TUI default — the
 *     terminal module must be used.)
 *  4. On exit, stops ONLY a launcher-owned server (`ollama stop <model>`
 *     + kill the spawned serve PID). A pre-existing server is left alone:
 *     the child gets `AGENT_STOP_OLLAMA_ON_EXIT=false` so risa's own
 *     shutdown hook doesn't kill it either.
 *
 * Modes: `--test` runs pytest instead of the TUI. `--check`, `--doctor`,
 * `--list-models` pass straight through. Anything after `--`
 * (`npm run ASCS -- --check`) is forwarded.
 *
 * Stdlib only (child_process, fs, http, os, path, url).
 */

import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { get } from "node:http";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import os from "node:os";

const ROOT = resolve(join(dirname(fileURLToPath(import.meta.url)), ".."));
const VENV_PYTHON = join(ROOT, ".venv", "Scripts", "python.exe");
// Overridable for testing (e.g. ASCS_OLLAMA_PORT=1 forces the down-path).
const OLLAMA_HOST = process.env.ASCS_OLLAMA_HOST ?? "127.0.0.1";
const OLLAMA_PORT = Number(process.env.ASCS_OLLAMA_PORT ?? 11434);
const PRIMARY_MODEL = "qwen3-coder:30b";
const FALLBACK_MODEL = "qwen2.5-coder:14b";
const SERVE_READY_TIMEOUT_MS = 60000;

function fail(message, code = 2) {
  console.error(`ascs: ERROR: ${message}`);
  process.exit(code);
}

/** Run a command with inherited stdio; return its exit code. */
function runForeground(cmd, args) {
  const result = spawnSync(cmd, args, { stdio: "inherit", cwd: ROOT });
  if (result.error) {
    fail(`could not run '${cmd}': ${result.error.message}`);
  }
  return result.status ?? 1;
}

/** Run a command quietly; return { status, error }. */
function runQuiet(cmd, args) {
  const result = spawnSync(cmd, args, {
    stdio: "pipe",
    encoding: "utf8",
    cwd: ROOT,
  });
  return { status: result.status ?? 1, error: result.error ?? null };
}

function ensureVenv() {
  if (existsSync(VENV_PYTHON)) {
    const probe = runQuiet(VENV_PYTHON, ["-c", "import agent"]);
    if (probe.status === 0) return; // healthy venv, nothing to do
    console.error(
      "ascs: venv python or 'agent' import is broken; reinstalling deps..."
    );
    const code = runForeground(VENV_PYTHON, [
      "-m",
      "pip",
      "install",
      "-r",
      "requirements.txt",
    ]);
    if (code !== 0) {
      fail(
        "pip install failed; run it manually: .venv\\Scripts\\python.exe -m pip install -r requirements.txt"
      );
    }
    return;
  }
  console.error("ascs: creating .venv ...");
  let created = false;
  for (const launcher of [
    ["py", "-3.12"],
    ["py", "-3"],
  ]) {
    const result = runQuiet(launcher[0], [
      ...launcher.slice(1),
      "-m",
      "venv",
      ".venv",
    ]);
    if (result.status === 0 && existsSync(VENV_PYTHON)) {
      created = true;
      break;
    }
  }
  if (!created) {
    fail(
      "could not create .venv (tried 'py -3.12' then 'py -3'). Install Python 3.12+ with the 'py' launcher and retry."
    );
  }
  runForeground(VENV_PYTHON, ["-m", "pip", "install", "--upgrade", "pip"]);
  const deps = runForeground(VENV_PYTHON, [
    "-m",
    "pip",
    "install",
    "-r",
    "requirements.txt",
  ]);
  if (deps !== 0) fail("pip install -r requirements.txt failed.");
}

function fetchJson(path) {
  return new Promise((resolvePromise, rejectPromise) => {
    const request = get(
      { host: OLLAMA_HOST, port: OLLAMA_PORT, path, timeout: 5000 },
      (response) => {
        let body = "";
        response.on("data", (chunk) => {
          body += chunk;
        });
        response.on("end", () => {
          try {
            resolvePromise(JSON.parse(body));
          } catch (error) {
            rejectPromise(error);
          }
        });
      }
    );
    request.on("timeout", () => {
      request.destroy(new Error("timed out"));
    });
    request.on("error", rejectPromise);
  });
}

async function ollamaUp() {
  try {
    await fetchJson("/api/tags");
    return true;
  } catch {
    return false;
  }
}

function sleep(ms) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms));
}

/**
 * Ensure an Ollama server is reachable. Returns `{ owned, servePid }`:
 * `owned` is true only when this launcher started `ollama serve` itself.
 * Never throws for a started-but-unready server — fails fast instead.
 */
async function ensureOllama() {
  if (await ollamaUp()) {
    return { owned: false, servePid: null };
  }
  console.error("ascs: Ollama is down; starting 'ollama serve' ...");
  const server = spawn("ollama", ["serve"], {
    detached: true,
    stdio: "ignore",
    cwd: ROOT,
  });
  let spawnError = null;
  server.on("error", (error) => {
    spawnError = error;
  });
  // Give spawn a tick to report ENOENT (missing binary) before polling.
  await sleep(500);
  if (spawnError) {
    fail(
      `could not spawn 'ollama' (${spawnError.message}). Install Ollama (https://ollama.com) and ensure it is on PATH, or start it with 'ollama serve' and retry.`
    );
  }
  if (server.exitCode !== null) {
    fail(
      `'ollama serve' exited immediately (code ${server.exitCode}). Start it manually with 'ollama serve' to see why, then retry.`
    );
  }
  server.unref();
  const deadline = Date.now() + SERVE_READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await sleep(1000);
    if (await ollamaUp()) {
      console.error("ascs: Ollama server is up (launcher-owned).");
      return { owned: true, servePid: server.pid };
    }
  }
  stopServeProcess(server.pid);
  fail(
    `'ollama serve' did not become ready within ${
      SERVE_READY_TIMEOUT_MS / 1000
    }s. Start it manually with 'ollama serve' to see why, then retry.`
  );
}

/** Best-effort kill of a serve PID we started. Never throws. */
function stopServeProcess(pid) {
  try {
    if (pid == null) return;
    if (os.platform() === "win32") {
      spawnSync("taskkill", ["/PID", String(pid), "/F"], { stdio: "ignore" });
    } else {
      process.kill(pid, "SIGTERM");
    }
  } catch {
    // best-effort only
  }
}

/** Best-effort unload of the active model. Never throws. */
function unloadModel(model) {
  try {
    spawnSync("ollama", ["stop", model], { stdio: "ignore", timeout: 60000 });
  } catch {
    // best-effort only
  }
}

async function checkModels() {
  const tags = await fetchJson("/api/tags");
  const names = new Set((tags.models ?? []).map((model) => model.name));
  const hasPrimary = [...names].some((name) => name.startsWith(PRIMARY_MODEL));
  const hasFallback = [...names].some((name) =>
    name.startsWith(FALLBACK_MODEL)
  );
  if (!hasPrimary && !hasFallback) {
    fail(
      `no usable model pulled. Run 'ollama pull ${PRIMARY_MODEL}' (or the 16GB fallback 'ollama pull ${FALLBACK_MODEL}'), then retry.`
    );
  }
  return hasPrimary ? PRIMARY_MODEL : FALLBACK_MODEL;
}

async function main() {
  const rawArgs = process.argv.slice(2);
  process.chdir(ROOT);
  ensureVenv();

  if (rawArgs.includes("--test")) {
    process.exit(runForeground(VENV_PYTHON, ["-m", "pytest", "-q"]));
  }

  const { owned, servePid } = await ensureOllama();
  const activeModel = await checkModels();

  const childEnv = { ...process.env };
  if (!owned) {
    // Pre-existing server: risa's shutdown hook must not kill it.
    childEnv.AGENT_STOP_OLLAMA_ON_EXIT = "false";
  }
  const child = spawn(VENV_PYTHON, ["-m", "agent.terminal", ...rawArgs], {
    stdio: "inherit",
    env: childEnv,
  });
  const shutdownOwned = () => {
    if (owned) {
      unloadModel(activeModel);
      stopServeProcess(servePid);
    }
  };
  child.on("error", (error) => {
    shutdownOwned();
    fail(`could not launch risa: ${error.message}`);
  });
  child.on("exit", (code, signal) => {
    shutdownOwned();
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 1);
  });
}

await main();
