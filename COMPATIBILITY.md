# ASCS Compatibility

Machine-readable source of truth: [`compatibility.json`](compatibility.json).

## Platforms

| OS | Architectures | Minimum | Status |
| --- | --- | --- | --- |
| Windows | x86_64, arm64 | 10 22H2 | CI smoke on windows-latest; Win10 needs hardware validation |
| macOS | x86_64, arm64 | 12.0 | CI smoke on macos-latest |
| Linux | x86_64, arm64 | glibc 2.28 | CI tests on ubuntu-latest |

## Runtime

- Python 3.12+, console scripts `ascs` and `risa`.
- pip, pipx and standalone executables.

## Model providers

`ollama`, `openai`, `anthropic`, `grok`, `google`, `deepseek`, plus
OpenAI-compatible custom endpoints. No provider is hard-coded.

## CORE integration

Optional. Requires device verification, authentication, server validation,
permission enforcement and protocol compatibility (`0.3.0`/`0.4.0`).
ASCS continues to run when CORE is unavailable.

## Validation levels

1. Source compilation and tests: verified.
2. Package generation: release workflow.
3. Installer: not tested on physical Windows hardware.
4. Runtime: macOS local tests and CI smoke on Linux/Windows/macOS.
5. Physical hardware validation: **not performed** for Windows 10/ARM64 or
   macOS Intel.
