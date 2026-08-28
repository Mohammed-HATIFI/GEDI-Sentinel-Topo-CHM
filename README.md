# Multisource canopy-height mapping across three contrasting Moroccan forest landscapes: aggregate error versus height-dependent performance under GEDI supervision

**GitHub repository title:** *Multisource canopy-height mapping across three contrasting Moroccan forest landscapes: aggregate error versus height-dependent performance under GEDI supervision*

**Associated Zenodo archive title:** *Derived data, trained models, and canopy-height maps for GEDI-Sentinel canopy-height mapping in Morocco*

Research code and reproducibility material for a two-stage, locally trained canopy-height mapping framework evaluated in Ifran, Maamoura, and Agadir, Morocco.

> **Release status.** The repository contains the source-code snapshot, final
> quality-filtered GEDI catalogues, frozen spatial partitions, exact evaluation
> supports, configuration records, and lightweight result tables. Raw GEDI
> granules, Sentinel imagery, derived tensors, trained checkpoints, and
> wall-to-wall GeoTIFFs are not stored in Git because of size and redistribution
> constraints. The associated Zenodo package is prepared as a private draft;
> its DOI will be added only after the record is published.

## Scientific workflow

1. Prepare monthly Sentinel-1, Sentinel-2, elevation, slope, AOI masks, and quality-filtered GEDI RH95 observations on a common 10 m grid.
2. Build distribution-informed spatial train/validation/test partitions.
3. Train the Phase 1 Attention U-Net for annual spatial canopy-height retrieval from 512 x 512 source patches.
4. Freeze Phase 1 and train/evaluate the Phase 2 four-year residual Conv3D head using 96 x 96 stochastic crops from complete four-year sequences.
5. Select all configurations on the spatial validation partition only.
6. Evaluate frozen products on held-out GEDI support, then generate annual maps and product-level comparisons.

The final Phase 2 selection uses residual-only refinement in all three landscapes (`lambda_temp = 0`). See [`configs/final_models.json`](configs/final_models.json).

## Quick start

```bash
git clone https://github.com/WiseManShady/GEDI-Sentinel-CHM.git
cd GEDI-Sentinel-CHM
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/check_release.py
pytest -q
```

Copy `configs/paths.example.yaml` to `configs/paths.local.yaml` and edit it for your storage layout. The local file is ignored by Git; never commit drive letters, credentials, or private GEDI registries.

## Repository layout

```text
configs/                 final model registry and portable path template
data/                    filtered GEDI data, evaluation supports, split layers, and manifest
docs/                    pipeline, data, provenance, and reproduction guide
models/                  checkpoint manifest (binaries live in the archive)
notebooks/official/      output-stripped audit/reproduction notebooks
results/metrics/         lightweight authoritative result tables
scripts/                 release, manifest, and notebook utilities
src/pipeline/            Phase 2, inference, catalogue, and audit snapshot
src/training/            Phase 1/Phase 2 training and loss snapshot
src/evaluation/          checkpoint ranking and evaluation snapshot
tests/                   smoke tests for metrics and release consistency
archive/initial_scaffold original placeholder scaffold kept for provenance
```

## Reproduction levels

- **Level 1 — audit:** verify configurations, hashes, filtered GEDI catalogues,
  evaluation supports, result tables, and notebook order without large rasters.
- **Level 2 — evaluation:** obtain the checkpoint/archive deposit, verify SHA-256
  hashes, and regenerate metrics and figures from the released supports.
- **Level 3 — full training:** acquire the upstream products, create derived monthly rasters and GEDI catalogues, then rerun both phases.

Detailed commands and expected artefacts are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Supplementary archive

The consolidated supplementary record is available as
[`docs/supplementary/Complete_Supplementary_Archive.pdf`](docs/supplementary/Complete_Supplementary_Archive.pdf).
Its inventory distinguishes included scientific analyses from empty historical
stubs and byte-identical duplicates. Focused audit notebooks are stored in
`notebooks/audits/`, while the main reproduction sequence remains in
`notebooks/official/`.

## Data availability

Upstream sources include NASA GEDI, Copernicus Sentinel-1/2, SRTM, ESA WorldCover, and Dynamic World. Derived inputs cannot all be committed to GitHub. The release uses:

- `data/manifest.csv` for logical file names, roles, checksums, sizes, and archive identifiers;
- `models/manifest.json` for final checkpoint identities;
- `data/split_summary.csv` for non-sensitive partition metadata;
- `data/processed/gedi/` for the final filtered shot/sample catalogues;
- `data/processed/evaluation/` for exact held-out and strict-common supports;
- `data/geospatial/` for portable spatial-partition layers;
- a versioned Zenodo archive for redistributable derived files, currently maintained as a private draft pending publication.

The empty split JSON files in the initial scaffold were removed because they were not valid reproduction artefacts.

## Source of truth

1. `configs/phase1_models.json` and `configs/final_models.json` — path-free final VAL-only model registries and hashes.
2. `results/metrics/` — frozen tables used by the manuscript.
3. `notebooks/official/README.md` — notebook execution order.
4. `models/manifest.json` and `data/manifest.csv` — external artefact identities.
5. `docs/PROJECT_ORGANIZATION.md` — distinction between the production workflow, audits, ablations, and supplementary evidence.

Historical scripts can retain execution-machine paths; those strings document provenance and are not portable defaults. New runs must use a local path configuration.

## Citation, license, and limitations

Citation metadata are in [`CITATION.cff`](CITATION.cff). Code is MIT licensed. Upstream data and third-party CHMs retain their own licenses.

- GitHub includes the processed GEDI registries, but not raw GEDI granules, raw
  imagery, tensor catalogues, checkpoints, or full annual CHMs.
- The spatial split is geographically held out but height-distribution informed; it does not estimate transfer to an arbitrary height distribution.
- External CHM comparisons are product-level comparisons, not controlled architecture comparisons.
- Archive DOIs and measured compute requirements remain release gates and must not be guessed.
