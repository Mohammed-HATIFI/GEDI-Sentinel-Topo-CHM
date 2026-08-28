# Project organization and source-of-truth rules

## Production workflow

The nine notebooks in `notebooks/official/` reproduce the frozen Phase 2
evaluation, article figures, dense inference, external-product comparisons,
height-class diagnostics, MSE decomposition, and residual-interval diagnostics.
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
evidence, not production entry points. They must select configurations on VAL
only. TEST outputs cannot be used for tuning. The paired Phase 1-to-Phase 2
multi-seed experiment remains an open audit until all matched-seed Phase 2 runs
are frozen and evaluated on the same TEST identifiers.

## Release levels

GitHub provides code, lightweight processed data, exact supports, and audit
tables. The Zenodo data/model archive provides checkpoints, annual GeoTIFFs,
training catalogues, spatial layers, and machine-readable lineage. Provider
controlled source rasters are never redistributed.
