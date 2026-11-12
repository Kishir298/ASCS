# ASCS Releases

A.S.C.S. versions independently. Current: `0.3.0` (`pyproject.toml`,
`compatibility.json`, `package.json`).

## Channels

| Channel | Trigger | Result |
| --- | --- | --- |
| Stable | tag `vX.Y.Z` | GitHub Release |
| Beta | tag `vX.Y.Z-beta.N` | prerelease |
| Development | `workflow_dispatch` | artifacts only |

## Artifacts

`ASCS-<version>-<os>-<arch>-<format>`:

- `ASCS-<v>-windows-x64.zip` (`ascs.exe`), `-windows-arm64.zip` (experimental)
- `ASCS-<v>-macos-x86_64.tar.gz`, `-macos-arm64.tar.gz`
- `ASCS-<v>-linux-x64.tar.gz`, `-linux-arm64.tar.gz`
- `ASCS-<v>-python-wheel.whl`, `-python-sdist.tar.gz`
- `SHA256SUMS`, `sbom.cdx.json`

## Process

1. Bump the version in `pyproject.toml`, `package.json`, `compatibility.json`
   and `CHANGELOG.md`.
2. Run the suite and `ruff check --select F agent tests`.
3. Commit; tag `vX.Y.Z`; push the tag.
4. The workflow validates, builds every target, verifies artifacts and
   publishes.
5. A failed required job blocks publishing.

## Limitations

- macOS signing/notarization requires configured secrets.
- Windows ARM64 is experimental.
- No native installers are produced; portable ZIP/tar archives are the
  supported Windows/macOS format.
