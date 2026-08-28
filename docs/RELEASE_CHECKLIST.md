# Public-release checklist

## Included and verifiable in Git

- [x] Path-free Phase 1 and Phase 2 model registries with SHA-256 hashes
- [x] Exact research-code snapshot and output-stripped official notebooks
- [x] Frozen lightweight metric tables
- [x] Frozen patch-to-split manifests for all three study areas
- [x] Final quality-filtered GEDI shot and sample catalogues for all three study areas
- [x] Exact Phase 2, external-product, and footprint-sensitivity evaluation supports
- [x] Portable GeoJSON spatial partitions and sampled-domain outlines
- [x] SHA-256 and byte-size manifest for all repository-included data files
- [x] Environment specifications and data-free CI checks
- [x] Data, model, provenance, and compute documentation

## Required before claiming full end-to-end reproducibility

- [x] Prepare selected checkpoints and permitted derived artefacts as a private Zenodo 1.0.0 draft
- [ ] Replace `TO_BE_ADDED` archive identifiers in `models/manifest.json`
- [x] Deposit the processed shot-to-split registries used by the final catalogues
- [x] Define archive licences (CC BY 4.0 for derived data; MIT for utilities)
- [ ] Add the published Zenodo DOI/URL to the repository and manuscript
- [ ] Record measured hardware, wall-clock time, and peak memory for the released runs
- [ ] Run Level 2 evaluation in a clean environment on a second machine

Until these gates are complete, describe the repository as a **code, processed-data,
and audit release**, not as a self-contained end-to-end training archive.
