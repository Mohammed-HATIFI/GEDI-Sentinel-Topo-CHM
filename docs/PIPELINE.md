# Pipeline map

| Stage | Main code | Inputs | Outputs |
|---|---|---|---|
| Preflight/config | `src/pipeline/b4_c15_preflight.py`, `b4_c15_config.py` | paths, channel contract | validated configuration |
| Catalogue construction | `src/pipeline/build_b4_c15_catalog.py` | aligned rasters, GEDI registry | training/evaluation catalogues |
| Phase 1 training | `src/training/phase1_train_catalog.py` | monthly 15-channel crops | Attention U-Net checkpoint |
| Phase 2 training | `src/pipeline/final_phase2_harmonized_workflow.py` | frozen Phase 1, four-year sequences | residual Conv3D checkpoint |
| Held-out evaluation | `src/pipeline/step07_evaluate_and_plot.py` | checkpoint, test support | metrics and figures |
| Annual inference | `src/pipeline/phase2_c15_inference_three_forests.py` | dense sequences | annual 10 m CHM |
| Temporal audit | `src/pipeline/temporal_coherence_audit.py` | Phase 1/2 predictions | coherence diagnostics |

Notebook orchestration is documented in `notebooks/official/README.md`.
