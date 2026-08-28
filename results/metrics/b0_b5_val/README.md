# Ifran B0-B5 validation audit

This bundle is the portable source of truth for the input-configuration screening reported in Appendix B.

- Selection used **VAL only** (`n=7,833`); TEST was not used to choose B4.
- All configurations used seed 42, Huber loss (delta 3 m), dropout 0.15, batch size 8, AdamW (`lr=1e-4`, weight decay `5e-3`), gradient clipping 1.0, `balanced_height` sampling, and `best.ckpt`.
- `tables/b0_b5_val_metrics.csv` is the canonical compact table.
- `B0_B5_VAL_Reproducibility.ipynb` regenerates the original validation audit table and diagnostic figure without retraining.
- `B0_B5_VAL_MSE_Decomposition.ipynb` reconstructs the exact VAL-only MSE decomposition from the archived common-support diagnostics.
- `tables/b0_b5_val_mse_decomposition.csv` is the numerical source for the VAL MSE-decomposition figure.
- `source_inventory.json` records the original local sources and SHA-256 hashes.
- `source_snapshots/` contains the small CSV evidence available for redistribution.

The notebook does not open TEST and does not retrain models. It rebuilds publication artefacts from frozen VAL diagnostics.
