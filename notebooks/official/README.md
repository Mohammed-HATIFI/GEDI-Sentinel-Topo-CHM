# Numbered publication notebooks

The notebooks follow the scientific order of the manuscript. Large embedded outputs are removed from the Git release; the authoritative lightweight tables and publication figures are stored under `results/metrics/` and `docs/supplementary/`.

1. `01_GEDI_Preprocessing_and_Catalog_Audit.ipynb` — audits GEDI preprocessing outputs, quality control, unique shots, and retained support.
2. `02_Study_Areas_Spatial_Splits_and_GEDI_Support.ipynb` — reproduces study-area maps, spatial splits, and GEDI distributions (Figs. 1–2; Table 1).
3. `03_Phase1_Spatial_Canopy_Height_Model.ipynb` — Phase 1 training and checkpoint provenance.
4. `04_Phase2_MultiYear_Residual_Refinement.ipynb` — frozen-backbone Conv3D residual refinement.
5. `05_Phase2_TEST_Diagnostics_and_Table2.ipynb` — Phase 2 TEST diagnostics and Table 2 (Figs. 5–7).
6. `06_Annual_CHM_Inference_and_Qualitative_Maps.ipynb` — annual inference, geospatial audit, and qualitative maps (Fig. 11).
7. `07_External_CHM_Benchmark.ipynb` — fixed-2020, strict-common four-product benchmark (Fig. 8; Table 3; Fig. 10 exports).
8. `08_Height_Stratified_MAE_and_Table_D1.ipynb` — rebuilds Table D.1, aggregate MAE, height-balanced MAE, upper-class sensitivity checks for Maamoura and Agadir, and Fig. 9.
9. `09_MSE_Decomposition_Appendix_D.ipynb` — MSE decomposition (Fig. D.1).

Executed validation, sensitivity, and ablation notebooks are numbered separately under `../Supplementary/`. Exploratory uncertainty analyses, alternative figure layouts, random qualitative-window proposals, backups, incomplete stubs, and duplicated cells are not part of the release sequence.
