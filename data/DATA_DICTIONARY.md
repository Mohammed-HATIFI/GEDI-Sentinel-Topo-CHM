# Data dictionary

## Final GEDI occurrence catalogues

Each row in `processed/gedi/<site>/shot_catalog_step05.csv.gz` is an eligible
GEDI occurrence paired with one model sample. A GEDI shot can therefore occur in
more than one row when it is eligible for several image dates.

Core fields:

- `split`: frozen spatial partition (`train`, `val`, or `test`).
- `site_id`, `patch_id`, `sample_id`, `patch_key`, `sample_key`: sample lineage.
- `aux_shot_uid`: stable GEDI-shot identifier used for de-duplication.
- `rh95`: filtered GEDI RH95 target in metres.
- `row`, `col`, `local_row`, `local_col`: raster and patch coordinates.
- `aux_gedi_date`, `aux_gedi_ordinal`: GEDI acquisition time.
- `aux_temporal_delta_days`, `aux_abs_temporal_delta_days`: GEDI-image time offset.
- `aux_lon`, `aux_lat`: coordinates when retained by the source catalogue.
- `aux_shot_weight`, `aux_height_weight`: train-only sampling/loss metadata.

## Evaluation tables

The three `*_phase2_test_unique_nearest.csv.gz` files contain one selected
occurrence per eligible held-out GEDI shot and the corresponding Phase 2
prediction. The selection is based on acquisition metadata, not the target value.

`external_chm_strict_common_support.csv.gz` stores shot-level GEDI RH95,
coordinates, product name, prediction, coverage, error, product year, and temporal
protocol. It is the authoritative input for the product-comparison tables and
height-class analyses.

`gedi_footprint_support_sensitivity.csv.gz` records predictions evaluated with
the alternative spatial-support operators used in the footprint sensitivity audit.

## Geospatial layers

`*_spatial_partitions.geojson` contains the frozen patch polygons and their split
labels. `*_sampled_domain.geojson` is their dissolved union. Coordinates and CRS
are stored in each GeoJSON; the layers are supplied for audit and visualisation.
