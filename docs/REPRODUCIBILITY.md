# Reproducibility guide

## Audit-only reproduction

```bash
python -m pip install -e .
python scripts/check_release.py
python scripts/verify_data_manifest.py
pytest -q
```

This checks repository structure, machine-readable manifests, the final model
registries, notebook metadata, included-data hashes, and absence of executable
placeholder claims. `results/metrics/artifact_manifest.csv` authenticates the
committed lightweight outputs.

## Evaluation reproduction

1. Verify the Git-included GEDI/evaluation data with
   `python scripts/verify_data_manifest.py`, then obtain the versioned checkpoint
   and raster archive listed in `data/manifest.csv` and `models/manifest.json`.
2. Copy `configs/paths.example.yaml` to `configs/paths.local.yaml`.
3. Configure archive roots and run `scripts/verify_artifacts.py`.
4. Execute the official notebooks in the documented order.

The test partition is evaluation only. Do not tune thresholds, checkpoints, architecture, or temporal weights after inspecting test metrics.

## Full training reproduction

### Phase 1

- Build monthly 15-channel catalogues (14 predictors plus AOI mask).
- Store source tensors as 512 x 512 x 15 arrays.
- Apply the frozen spatial split.
- Train the Attention U-Net with the site-specific Phase 1 Huber setting.
- Select checkpoints using validation diagnostics only.

### Phase 2

- Freeze the selected Phase 1 model.
- Construct complete four-year sequences.
- Draw 96 x 96 stochastic crops from the 512 x 512 source tensors; the model input crop contract is 4 x 15 x 96 x 96.
- Train the residual Conv3D head using `configs/final_models.json`.
- Keep `lambda_temp = 0` for Ifran/Maamoura and `0.10` for Agadir in the released selection.

### Evaluation/inference

- Evaluate GEDI on the frozen unique-nearest support.
- Generate annual maps using all complete contributing windows/months.
- Harmonise external CHMs and apply exact common support.

Entry points are mapped in `docs/PIPELINE.md`. Historical scripts may contain Windows execution paths; configure local paths without changing scientific parameters.

## Release gates

- archive URL/DOI and all SHA-256 checksums completed;
- exact split registry released (included under `data/processed/gedi/`);
- measured GPU/CPU/RAM/storage record completed in `docs/COMPUTE.md`;
- clean execution from a fresh environment;
- no notebook output depends on an untracked local file.
