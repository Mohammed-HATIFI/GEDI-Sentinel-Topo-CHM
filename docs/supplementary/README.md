# Complete supplementary archive

This folder preserves the complete supplementary scientific record associated with the Article 1 canopy-height manuscript at the archive date (23 August 2026).

## Primary file

- `Complete_Supplementary_Archive.pdf`: single 34-page archival PDF.

## Rebuild source

- `Complete_Supplementary_Archive.tex`: LaTeX master used to assemble the PDF.
- The master references the live project at:
  `C:\Users\Dell\Desktop\Publication_Clarck\Natural_Sampling\Writing_Article\1_Article\1_Article_RSE_Hybrid_Appendices`

To rebuild, run from that project directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error Complete_Supplementary_Archive.tex
```

## Scope

The PDF consolidates unique scientific content rather than every byte-level backup. It includes:

- preprocessing and GEDI filtering;
- B0--B5 input configurations, historical diagnostics, VAL diagnostics, and raw metric snapshots;
- Phase 1 architecture, training, and checkpoint selection;
- external-product provenance, strict-common scatter diagnostics, MSE decomposition, signed errors, class-wise MAE, and qualitative maps;
- Phase 2 workflow, temporal-weight tests, design controls, pixel-year collision sensitivity, clipping checks, footprint operators, patch replication, and multi-seed audits;
- final-model distribution/scatter diagnostics;
- post-hoc uncertainty diagnostics, explicitly retained as archival rather than independent validation.

## Exclusions

Compiler auxiliaries, intermediate manuscript PDFs, `.bak` copies, and byte-identical PNG/SVG counterparts of included PDF figures are not repeated. Placeholder-only supplementary shells containing comments but no results are recorded in the inventory and are not rendered as empty pages.

## Interpretation

Some retained figures are historical post-selection or post-hoc diagnostics. Their captions and section text distinguish them from validation-only selection evidence and from independent external validation.
