# Project organization and source-of-truth rules

## Production workflow

The nine numbered notebooks in `notebooks/official/` follow the manuscript
workflow from the GEDI preprocessing audit through spatial modelling,
multi-year refinement, held-out evaluation, dense inference, external-product
comparisons, height-class diagnostics, and MSE decomposition.
Their order is defined in `notebooks/official/README.md`.

## Model selection and authoritative results

- Phase 1 and Phase 2 model identities: `configs/phase1_models.json` and
  `configs/final_models.json`.
- Frozen numerical claims: `results/metrics/`.
- Exact evaluation supports: `data/processed/evaluation/`.
- Spatial partitions: `data/geospatial/` and `data/splits/`.
- Large checkpoint and map identities: `models/manifest.json` and
  `data/manifest.csv`.

## Audits and ablations

Controlled ablations and robustness notebooks are scientific supporting
evidence, not production entry points. The executed analyses retained for the
manuscript are numbered in `notebooks/Supplementary/`. Configuration decisions
must use VAL only; TEST outputs cannot be used for tuning. Incomplete,
superseded, and purely exploratory notebooks are excluded from the release.

## Release levels

GitHub provides code, lightweight processed data, exact supports, and audit
tables. The Zenodo data/model archive provides checkpoints, annual GeoTIFFs,
training catalogues, spatial layers, and machine-readable lineage. Provider
controlled source rasters are never redistributed.
