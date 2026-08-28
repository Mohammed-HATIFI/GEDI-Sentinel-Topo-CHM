# Release audit

Audit date: 2026-08-18

## Checks completed on the local repository

- `python scripts/check_release.py`: passed.
- All Python files parsed successfully with `ast.parse`.
- All official notebooks parsed as valid notebook JSON.
- The two metric unit tests passed when the local `src` directory was placed on
  `PYTHONPATH`.
- `git diff --check`: no whitespace errors (Git only reported the local Windows
  LF-to-CRLF warning).
- No tracked release candidate file exceeds 50 MiB.
- The repository contains nine output-stripped official notebooks.
- The three patch-to-split manifests are included with SHA-256 identities in
  `data/manifest.csv`.
- The six selected checkpoint hashes and byte sizes were verified against the
  original research workspace and recorded in `models/manifest.json`.

## Re-run locally

```powershell
python -m pip install -e .
python scripts/check_release.py
python -m pytest -q
```

After downloading the six checkpoint files from the future archive:

```powershell
python scripts/verify_artifacts.py `
  --manifest models/manifest.json `
  --root C:\path\to\downloaded\checkpoints
```

## Remaining publication gates

The Git repository is an audit-and-code release. A full end-to-end reproduction
claim still requires:

1. publication of the prepared private Zenodo 1.0.0 archive containing the
   selected checkpoints, derived maps, and permitted catalogues;
2. replacement of `TO_BE_ADDED` in `models/manifest.json` with the archive DOI;
3. final verification of the deposited split registries and portable paths;
4. measured hardware, wall-clock, and peak-memory records;
5. a clean Level 2 evaluation on a second machine.

These missing release artefacts are deliberately visible rather than replaced by
placeholder data or synthetic outputs.
