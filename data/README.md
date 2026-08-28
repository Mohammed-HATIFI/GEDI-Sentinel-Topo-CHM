# Data release and contract

This repository includes the final quality-filtered GEDI catalogues used by the
pipeline, the exact held-out evaluation-support tables used for the reported
metrics, and portable spatial-partition layers. It intentionally does not
redistribute the large upstream or derived rasters.

The 14 predictors are stored in a 15-channel tensor because the AOI mask is carried as an additional non-predictor mask channel.

## Included

- `processed/gedi/<site>/shot_catalog_step05.csv.gz`: final filtered GEDI
  occurrence catalogue with frozen split labels and RH95 targets.
- `processed/gedi/<site>/sample_catalog_step05.csv`: final sample catalogue.
- `processed/evaluation/*_phase2_test_unique_nearest.csv.gz`: exact Phase 2
  held-out supports and predictions.
- `processed/evaluation/external_chm_strict_common_support.csv.gz`: shot-level
  product comparison table.
- `processed/evaluation/gedi_footprint_support_sensitivity.csv.gz`: centre-pixel,
  3 x 3, and circular-support audit table.
- `geospatial/*_spatial_partitions.geojson`: frozen train/validation/test patch
  polygons.
- `geospatial/*_sampled_domain.geojson`: dissolved envelope of those sampled
  patches. These are sampled-domain outlines, not administrative or legal forest
  boundaries.

The duplicate pre-step05 catalogues are deliberately omitted. Their inclusion
would add size without improving reproducibility.

## Not included on GitHub

Full reproduction of training still requires monthly Sentinel-1 and Sentinel-2
rasters, SRTM elevation/slope, the AOI masks, derived tensor arrays, checkpoints,
annual CHMs, and external CHM rasters. These files are too large for ordinary Git
and remain provider downloads or research-archive deposits. They must be linked
from `manifest.csv` when the Zenodo/OSF archive is published.

The 14 predictors are stored in a 15-channel tensor because the AOI mask is carried
as an additional non-predictor mask channel.

See `DATA_DICTIONARY.md` for field-level interpretation and `manifest.csv` for
hashes and redistribution status.

## GEDI QC provenance

See `../docs/GEDI_FILTERING_PROVENANCE.md` and `../notebooks/audits/GEDI_STEP05_Catalog_Audit.ipynb`. The release reproduces training from the frozen post-QC catalogues; the original upstream raw-GEDI filtering notebook was not recovered.
