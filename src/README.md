# Source-code status

`gedi_sentinel_chm/` is the portable, tested Python package. The `pipeline/`,
`training/`, and `evaluation/` directories preserve the exact research-code snapshot
used for the experiments. Some snapshot modules retain historical absolute paths;
they are provenance records, not portable defaults. Public reproduction should use
`configs/paths.local.yaml` (copied from `configs/paths.example.yaml`) and the wrappers
documented in `docs/REPRODUCIBILITY.md`.
