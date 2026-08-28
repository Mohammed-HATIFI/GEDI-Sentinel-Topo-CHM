# GEDI filtering and STEP05 catalogue provenance

## Scope

The released `shot_catalog_step05.csv.gz` files are the exact post-QC,
post-patch-construction occurrence catalogues consumed by the training pipeline.
The accompanying STEP05 workbooks provide one row per unique GEDI shot within
each frozen spatial split. They do not replace the occurrence catalogues used by
the dataloader.

## Archived quality-control contract

Before catalogue construction, GEDI L2A footprints were restricted to the exact
study polygons and screened for sensor/waveform quality, physical consistency,
sensitivity, and terrain slope. The documented rules were:

- valid `quality_flag` and non-degraded acquisition;
- positive `num_detectedmodes`, finite RH95, and valid waveform/algorithm flags;
- finite elevation variables and non-negative waveform extent;
- non-negative RH95 and RH95 no more than 2 m above waveform-derived extent;
- duplicate removal by granule, beam, and shot number;
- sensitivity at least 0.90;
- terrain slope at most 20 degrees;
- a site-specific nighttime criterion (`solar_elevation < 0`) for Maamoura.

No fixed upper canopy-height threshold was applied during GEDI quality control.
The evaluation ranges were applied later to the frozen TEST supports.

## Exact STEP05 unique-shot counts

| Site | Train | Validation | Test | Total unique shots |
|---|---:|---:|---:|---:|
| Ifran | 42,204 | 8,856 | 5,190 | 56,250 |
| Maamoura | 5,938 | 1,312 | 1,870 | 9,120 |
| Agadir | 62,707 | 11,870 | 9,474 | 84,051 |

These counts were re-read directly from the exact frozen STEP05 catalogues.
The Maamoura value is 9,120; older workbooks/drafts reported 8,384 or 9,136
from different catalogue lineages and must not be used for the final release.

## Reproducibility boundary

The current project workspace contains the final filtered catalogues and the
catalogue-building/training pipeline, but the original notebook or script that
performed the upstream L2A-to-QC filtering was not recovered. Consequently,
the release supports exact reproduction from the frozen filtered catalogues,
not byte-for-byte regeneration from raw GEDI granules. This boundary is stated
explicitly to avoid overstating provenance.

`GEDI_STEP05_Catalog_Audit.ipynb` verifies the released catalogue counts,
split labels, unique-shot keys, RH95 ranges, and hashes without modifying the
archived measurements.
