# Official notebook order

Large cell outputs are stripped from the Git release. Authoritative lightweight tables are kept in `results/metrics/`.

1. `Study_Area.ipynb` — study-area, split and GEDI-support figures.
2. `Generate_Figure2_Redesigned.ipynb` — condensed study-area figure.
3. `Phase_2.ipynb` — evaluate selected checkpoints; retraining disabled.
4. `Graphs_Model_Phase_2.ipynb` — final-model plots and height diagnostics.
5. `Phase1_Phase2_Temporal_Coherence.ipynb` — paired Phase 1--Phase 2 diagnostics.
6. `Inference.ipynb` — annual inference and geospatial audit.
7. `Inference_Forest_By_Forest.ipynb` — reproducible qualitative-region candidates.
8. `Chm_Comparison.ipynb` — strict-common external-product benchmark.
9. `CHM_Height_Class_Comparison_Strict_Common.ipynb` — product comparison by height class.
10. `MSE_Decomposition.ipynb` — bias/spread/correlation decomposition.
11. `CHM_Conformal_Uncertainty_Comparison.ipynb` — exploratory interval diagnostics.

Focused selection and robustness notebooks are stored separately in
`../audits/`. All public notebook copies are output-stripped; authoritative
tables and publication figures are retained under `results/metrics/` and
`docs/supplementary/`.

0. `Phase_1.ipynb` — Phase 1 training workflow and checkpoint provenance (output-stripped).
