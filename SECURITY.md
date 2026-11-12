# ASCS Security

## Reporting

Report vulnerabilities privately to the repository owner.

## Implemented controls

- API keys are read from the environment/runtime configuration, never written
  to logs or persisted in the repository.
- Workspace mutations are path-normalized and restricted to the configured
  workspace; pre-existing dirty files are preserved.
- Verification commands require real actionable checks; informational notes
  do not count as verification.
- Controlled command execution with timeouts, cancellation and clean process
  termination.
- CORE credentials are optional and runtime-only; CORE-disabled operation is
  the default.
- Release artifacts carry SHA-256 checksums and an SBOM.
- GitHub Actions use least-privilege permissions.

## Secrets

Do not commit `.env`, API keys, `*.pem`, `*.key` or device files. Use dummy
credentials in tests.

## Known limitations

- The local Ollama endpoint is unauthenticated by default; bind it to
  loopback.
- Standalone binaries are unsigned unless signing is configured.
- `--safe` approval overlays are advisory; review diffs before running in
  fully autonomous mode.
