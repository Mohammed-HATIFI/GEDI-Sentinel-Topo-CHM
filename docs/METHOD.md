# Method contract

This file records the computational contract. The manuscript remains the source for
the scientific rationale and citations.

## Predictor tensor

Each annual source patch has shape `512 x 512 x 15`. Fourteen predictor layers are
used by the model:

- Sentinel-2 B02, B03, B04, B08, B05, B06, B07, and B8A;
- Sentinel-1 ascending VV/VH and descending VV/VH;
- SRTM elevation and slope.

The fifteenth layer is the AOI mask. It controls support and is not described as an
Earth-observation predictor. PALSAR and Sentinel-2 SWIR are not part of the retained
B4 input configuration.

## Spatial partitions

Train, validation, and test patches are geographically disjoint and retain unique GEDI
shot assignments. The split construction is distribution-informed. Target fractions
belong to the catalogue provenance: the rebuilt B4 catalogue uses 75/15/10, while the
legacy Agadir catalogue records 70/15/15. Model and hyperparameter selection use the
spatial validation partition only; test observations are evaluation only.

## Phase 1

Phase 1 is an Attention U-Net supervised by quality-filtered GEDI RH95 using a masked
Huber loss. It consumes the annual 15-layer source patch, predicts a dense 10 m canopy-
height field, and uses the site-specific Huber settings recorded in
`configs/phase1_models.json`.

## Phase 2

The Phase 1 backbone is frozen. Complete same-month four-year sequences are formed
from 512 x 512 source tensors, and training draws `4 x 15 x 96 x 96` crops. A residual
Conv3D head refines the Phase 1 sequence. The released product uses residual-only refinement in all three landscapes
(`lambda_temp = 0`). Four-year sequences balance temporal context against the
availability of complete same-patch, same-month, GEDI-anchored samples; longer
sequences would reduce the eligible training support. Breakpoint and Huber settings are recorded in
`configs/final_models.json`.

## Evaluation

The frozen selections are evaluated on unique-nearest held-out GEDI support. Reported
diagnostics include MAE, RMSE, R2, correlation, bias, regression slope, prediction-to-
observation spread ratio, KGE where applicable, height-class errors, and MSE
decomposition. External CHMs are compared on identical common GEDI support. This is a
product-level comparison, not a controlled comparison of architectures.
