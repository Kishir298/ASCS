#!/usr/bin/env node
/**
 * One-command A.S.C.S. launcher: `npm run ASCS`.
 *
 *  1. Ensures `.venv` exists (creates with `py -3.12`, falls back to
 *     `py -3`) and `requirements.txt` is installed; reinstalls only when
 *     the venv python or the `agent` import is broken.
 *  2. Checks Ollama is reachable at localhost:11434 and that
 *     `qwen3-coder:30b` (or the `qwen2.5-coder:14b` fallback) is pulled.
 *     Fails fast with the exact fix command — never silently pulls GBs.
 *  3. Launches `.venv\Scripts\python.exe -m agent <args>` with inherited
 *     stdio (required for the curses TUI), forwarding the exit code. No
 *     args → TUI via the existing `normalize_argv` default.
 *
 * Modes: `--test` runs pytest instead of the TUI. `--check`, `--doctor`,
 * `--list-models` pass straight through to risa. Anything after `--`
 * (`npm run ASCS -- --check`) is forwarded to `python -m agent`.
 *
 * Stdlib only (child_process, fs, http, path, url).
 */

import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { get } from "node:http";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(join(dirname(fileURLToPath(import.meta.url)), ".."));
const VENV_PYTHON = join(ROOT, ".venv", "Scripts", "python.exe");
// Overridable for testing (e.g. ASCS_OLLAMA_PORT=1 forces the down-path).
const OLLAMA_HOST = process.env.ASCS_OLLAMA_HOST ?? "127.0.0.1";
const OLLAMA_PORT = Number(process.env.ASCS_OLLAMA_PORT ?? 11434);
const PRIMARY_MODEL = "qwen3-coder:30b";
const FALLBACK_MODEL = "qwen2.5-coder:14b";

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

async function checkOllama() {
  let tags;
  try {
    tags = await fetchJson("/api/tags");
  } catch {
    fail(
      `Ollama is not reachable at http://${OLLAMA_HOST}:${OLLAMA_PORT}. Start it with 'ollama serve', then retry.`
    );
  }
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
}

async function main() {
  const rawArgs = process.argv.slice(2);
  process.chdir(ROOT);
  ensureVenv();

  if (rawArgs.includes("--test")) {
    process.exit(runForeground(VENV_PYTHON, ["-m", "pytest", "-q"]));
  }

  await checkOllama();

  const child = spawn(VENV_PYTHON, ["-m", "agent", ...rawArgs], {
    stdio: "inherit",
  });
  child.on("error", (error) => fail(`could not launch risa: ${error.message}`));
  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 1);
  });
}

await main();
