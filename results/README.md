# Result artefacts

`metrics/` contains the lightweight frozen CSV/JSON outputs used by the manuscript.
These files are committed so that tables and summary claims can be audited without
downloading imagery or checkpoints. `artifact_manifest.csv` records their SHA-256
checksums. Large prediction registries, GeoTIFFs, and checkpoints belong in the
external research archive and are referenced by `data/manifest.csv` and
`models/manifest.json`.
