# ASCS update from the RISARMS server plan

6 November 2026. Baseline: Kishir298/ASCS commit 5e3889360c17ee9ca6100b0b087e37430e7bbe48. Input: RISARMS-Portal-Setup-and-Changes.docx, especially its final ASIS and ASCS on the Raspberry Pi chapter. This is an experimental agent update, not a validated Pi deployment. The upstream distribution version remains 0.3.0; identify this build by the dated archive and SHA256 manifest.

## Implemented behavior

The installed `ascs` command now exists alongside `risa`. Non-Pi use retains the terminal interface and existing agent behavior. Linux ARM64 automatically takes the protected Pi path; `--pi` or ASCS_PI_MODE=1 selects it explicitly for other Linux machines. `python -m agent.main` and the terminal entry point take the same path. Direct AgentLoop and TaskExecutor construction also rejects Pi full-agent execution. This prevents model-requested tools from executing through those agent engines. It is not an OS sandbox or protection against someone modifying/running arbitrary Python code themselves.

Pi operation generates text/code only. It never reads a repository into the prompt, changes project files, invokes generated tools or runs generated code. Plan/Build/Auto from an old CORE launcher do not bypass this restriction. Generated suggestions must be reviewed and applied separately. Full autonomous coding remains blocked until an enforceable OS sandbox and model tool protocol have been verified on ARM64.

Before generation, the adapter checks the exact installed tag and digest against an operator approval manifest, verifies Q4/Q5 quantization and chat metadata, rejects cloud-backed tags, then samples CPU for ten seconds and reads MemAvailable, cgroup v2 parent limits/current usage, memory PSI, temperature and system load. It rechecks admission before inference. Missing measurements fail closed. Swap is not counted. The manifest supplies a measured/conservative peak budget; the adapter subtracts a 1.5 GiB reserve and credits already resident model bytes once rather than counting them as a second allocation.

Initial filters follow the supplied document: 0.5B-class requires at least 2.5 GiB, 1.5B-class 3.5 GiB, and 3B-class 5 GiB. The peak-budget check can require more RAM than these initial filters. More than 3.2 billion reported parameters is denied; this admits common rounded 3B labels but excludes 4B/7B+ without an alternate approved implementation. It does not estimate parameter count from a name. Context is 2048 by default, or 4096 only when that exact context appears in the operator's verified manifest. Requests explicitly set two inference threads and 1024 maximum output tokens.

CPU averages over 70%, excessive load (over 1.5 times detected core count), memory PSI avg10 over 10%, or temperature at/above 75 C defer generation. Load/PSI/thermal thresholds are conservative implementation choices, not measured Pi safety guarantees. Current firmware undervoltage/throttle flags are NOT measured by this implementation; hardware validation must additionally check these. A temperature reading alone does not certify an unthrottled Pi.

A Linux flock slot permits one cooperating inference operation at a time. The administrator must provision the lock file; the application does not create a world-writable queue. Busy callers wait at most 30 seconds, then return exit 75. Resource deferral also returns 75 and structured JSON reasons. This is bounded mutual exclusion/deferral, not a persistent FIFO or automatic retry queue. CORE must display/handle this state; its old adapter otherwise labels a nonzero exit as failed.

Model switching never happens silently. A different resident model normally blocks. `--pi-unload-previous` explicitly permits unloading only resident tags whose digests are also present in the same approval manifest. Unload occurs under the shared slot and is confirmed with /api/ps before proceeding. Unapproved/unrelated residents are never unloaded. No process-name killing is used by the Pi adapter. keep_alive=0 requests release of the selected model after the response. SIGTERM/Ctrl+C releases the slot and exits 130; the Ollama daemon may finish an already submitted HTTP generation after client cancellation. Daemon parallelism limits remain necessary. No partially performed project action is retried because Pi mode performs none.

No model is downloaded, no cloud account is required, no laptop is required, and no public endpoint is published by this path. The model API must be loopback. HTTP redirects and proxy configuration are ignored/refused. The Ollama service drop-in additionally disables its cloud features. Logs/status include admission metadata; prompt and model text go only to the invoking caller's stdout, which CORE must scope to its authenticated account/job.

## Installation

ASCS still requires Python 3.12 or newer. A Pi OS image with Python 3.11 does not satisfy that requirement; provision a supported interpreter separately. Do not replace the system interpreter or assume CORE's lower Python minimum covers ASCS.

Run as a dedicated unprivileged account. Pi mode refuses effective UID 0. Provision that account and a trusted `risarms-ai` group with OS administration tools. It must not have sudo, Docker socket access, other accounts' data or raw RESCS directories. Install in that account's environment:

```sh
python3.12 -m venv /absolute/path/ASCS/.venv
/absolute/path/ASCS/.venv/bin/python -m pip install /absolute/path/ASCS
export PATH="/absolute/path/ASCS/.venv/bin:$PATH"
ascs --help
```

Add the PATH line to the service environment or the account's shell startup file to persist it. A normal non-editable installation remains usable from other directories. For Windows/macOS standalone installation, use the same Python minimum and install the package into an environment whose Scripts/bin directory is on PATH. Windows terminal UI support requires the `tui` extra. Pi runtime is Linux-only; native Windows/macOS behavior is not newly certified by this update.

Provision /run state at boot by installing deploy/pi/risarms-inference.conf under /etc/tmpfiles.d and running systemd-tmpfiles after the trusted worker group exists. The directory is root-owned, mode 0750; the existing lock is root:risarms-ai, mode 0660. Workers need group access. Do not put the lock in a private per-account directory: ASIS and ASCS must coordinate on the SAME file.

Install deploy/pi/models.example.json as /etc/risarms/ascs-models.json, owned by an administrator, readable by the worker and not writable by public accounts. The example deliberately cannot authorize a job. Replace its placeholder key with an exact installed instruct/quantization tag, record the actual digest, quantization, conservative peak GiB and tested contexts, and set generation_verified=true only after a supervised benchmark. For the first benchmark, use the operator's separate controlled Ollama request, not an unverified autonomous agent. No example model is certified merely by this release.

```sh
ascs --pi --pi-status --workspace /srv/core/ascs/projects/demo --model EXACT_APPROVED_TAG
ascs --pi --workspace /srv/core/ascs/projects/demo --model EXACT_APPROVED_TAG -- "Explain a short Python function"
# Explicit switch of approved previous models only:
ascs --pi --pi-unload-previous --workspace /srv/core/ascs/projects/demo --model EXACT_APPROVED_TAG -- "Suggest a small code snippet"
```

Workspace must exist. Status needs the manifest, provisioned lock, installed model and local runtime; it is not an offline hardware-only doctor. Two ten-second resource samples precede generation. Both initial status and deferral reasons are JSON; generation output is clearly labelled as text-only. Exit 0 means status allowed or a complete text response, 1 means blocked/error, 75 means transient deferral, 130 means cancellation. No TUI or input prompt opens on the Pi path. Legacy npm Pi invocation delegates directly to this runner and requires a prepared .venv; it does not install dependencies or start Ollama. The standard installed command does not require Node.js.

Configuration: ASCS_PI_MODE=1 enables protected mode; ASCS_PI_MANIFEST overrides the manifest path; RISARMS_INFERENCE_LOCK overrides the shared lock. These are trusted operator configuration, not browser-supplied values. The base URL can be explicitly set only to the supported loopback addresses on port 11434. An optional remote-model service is not implemented in Pi mode.

For offline installation, on a connected machine matching target Python/OS/architecture, build the package wheel and prepare any optional dependency wheels with `python -m pip wheel --no-deps . -w wheelhouse` and `python -m pip download --only-binary=:all: -d wheelhouse setuptools wheel pytest`. Transfer the wheelhouse; install with `python -m pip install --no-index --find-links wheelhouse coding-agent`. The supplied wheel has no base third-party runtime dependencies. Model weights, Python, Ollama and project toolchains remain separate preparations. Install/download them explicitly before disconnecting. Record exact versions/digests and keep the prior package for rollback.

## CORE and ASIS integration

integration/CORE-HOST-Pi-ASCS.patch targets the `core/web/services.py` source read from GitHub, blob 61fc9c86648b64dae062b48f74a5f0fa84d299ea. It sets ASCS_PI_MODE=1 for ASCS jobs, retaining CORE's owner-only authorization, profile workspace, experience path and noninteractive contract. Check it against the actual newer portal release before applying with `git apply --check`; no newer CORE ZIP was supplied with the document. Do not replace a newer service file with an old complete file.

The existing launch contract is preserved:

```text
python -u -m agent.main --workspace PROFILE_WORKSPACE --mode plan|build|auto -- TEXT
```

CORE must pass the exact approved --model tag. No default 30B model is selected in Pi mode. Member messaging remains tool-free in CORE and is not routed into this privileged coding adapter. Owner role checks remain CORE's responsibility. The JSON status is a dashboard integration surface; this ASCS update does not implement the newer owner dashboard itself or change public registration, RESCS ownership, file routes, Our Library or Day Quest.

The supplied `deploy/pi/risarms-inference` wrapper lets OTHER managed model callers (including ASIS and CORE's tool-free messaging worker) acquire the same lock using flock. ASCS acquires it internally: do not wrap ASCS again or it will wait on its own lock. The ASIS/public worker must also implement compatible admission/context limits before the combined deployment can claim coordinated resource safety. Their repositories were not changed in this ASCS task. Locking cannot constrain uncooperative programs or an exposed model API. Restrict access and use one dedicated managed runtime.

CORE's current subprocess adapter runs jobs as CORE's OS account. The one-line patch does NOT install an isolated worker broker, switch Unix identity or remove that account's filesystem access. Configure a separate worker/broker and test OS permissions before enabling full-agent code execution. Until then this package allows only text generation on Pi. Do not claim workspace validation is a container sandbox.

The Ollama drop-in is an operator-reviewed starting template, not an installed service: one loaded model, one generation, explicit 2048 context, local-only API, SSD model directory, Nice=10, CPUQuota=200%, MemoryHigh=4G and MemoryMax=5G. CPUQuota is a budget across cores, not a benchmark. Validate installed Ollama/systemd versions and adjust measured budgets. Ensure /srv/core is really mounted before starting data/model services; add and validate RequiresMountsFor=/srv/core in deployment. Do not use an SSD path on the microSD when the mount is missing. This package does not format drives, modify fstab or install service users.

## Verification and limitations

Final deterministic suite: 991 passed, 6 skipped, exit 0. Ruff F checks, compilation, workflow YAML parsing, wheel build and installation into a fresh environment outside the source directory passed. See validation/pytest-offline.log and XML for the final suite. The suite runs on Linux x86-64, Python 3.12, with a clean environment, external networking blocked and loopback allowed. Model responses and resource measurements are controlled fixtures. No actual Pi model inference, model download, ARM64 native execution, systemd installation, SSD failure, thermal-throttling measurement, multi-site latency or persistent scheduling benchmark has been performed.

The tests exercise tier gates, cgroup limits, reserve/residency accounting, CPU/memory/thermal deferral, queue exclusion/release, exact digest/quantization admission, cloud/remote rejection, explicit approved-model switching, unrelated resident protection, unsupported contexts, cancellation, no tool protocol and full-agent guards. Existing deterministic tests remain included. Live tests remain opt-in and were skipped. CI's Python matrix now matches >=3.12 and the invalid secrets-in-job-if workflow is replaced by explicit manual dispatch on a provisioned self-hosted Ollama runner. Ruff issues were resolved with preserved legacy re-exports and unused-local renames; no tests were removed to hide failures.

An initial unrestricted full suite was blocked by automatic review when an existing model-provider listing attempted an external request. That attempt is not a pass; subsequent full-suite evidence uses validation/offline_guard.py and no credentials. Hardware/network privacy must still be tested at the destination. No GitHub write succeeded: the connector returned HTTP 403 for branch creation. No public deployment occurred.

Compatibility matrix: Linux x86-64 package installation and deterministic tests executed; Linux ARM64 Pi policy branches simulated/experimental; Windows/macOS retained but not executed; full Pi tool execution blocked; owner dashboard and cross-service coordination require CORE/ASIS work; external remote inference disabled; private data access remains an OS/backend responsibility. No throughput or model-quality claim is made.

## Backups and rollback

Stop jobs and retain the old package/source, CORE service file, workspaces including .ascs state, per-profile settings/history and matching CORE/RESCS backups. This update does not migrate user data or merge accounts. Restore the prior source/wheel and CORE patch only while jobs are stopped. Restore matching data backups if deployment configuration changed. Do not remove data as part of uninstall. Rolling back to the old ASCS also restores its previous process/tool behavior; do not enable that full agent on the shared Pi without containment.

References checked: https://docs.ollama.com/faq (context, residency, parallelism, local-only configuration); https://github.com/ollama/ollama/blob/main/docs/openapi.yaml (tag/show/ps/chat metadata). These document the protocol, not performance of this release. The attached planning document is the authority for the selected Pi policy.
