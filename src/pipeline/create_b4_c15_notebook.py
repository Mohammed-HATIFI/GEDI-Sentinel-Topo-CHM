from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def forest_cells(site: str, chapter: int, label: str, historical: bool = False) -> list[dict]:
    upper = site.upper()
    native_note = (
        "Le catalogue historique contient 17 canaux stockés, mais le modèle B4 retire uniquement "
        "PALSAR aux indices 12 et 13 : l'entrée effective est donc **C15**. Aucun réentraînement n'est autorisé."
        if historical
        else "Le catalogue est construit directement en **C15 natif**, sans créer ni injecter de canal PALSAR."
    )
    cells = [
        markdown(
            f"""
# Chapitre {chapter} — Forêt de {label}

{native_note}

Ordre contractuel : 8 bandes Sentinel-2 (`B02, B03, B04, B08, B05, B06, B07, B8A`) +
4 canaux Sentinel-1 (`ASC VV/VH`, `DESC VV/VH`) + `AOI_MASK` + `DEM` + `SLOPE` = **15 canaux**.
"""
        ),
        markdown(f"## {chapter}.1 — Chemins, contrat C15 et préflight"),
        code(
            rf"""
site = SITES["{site}"]
print("Forêt       :", site.label)
print("Catalogue   :", site.catalog_root)
print("Run         :", site.run_dir)
print("S2          :", site.s2_dir)
print("DEM         :", site.dem)
print("Pente       :", site.slope)
print("Canaux C15  :", B4_C15_CHANNEL_ORDER)
assert_c15_contract(B4_C15_CHANNEL_ORDER)

report_{site} = run_preflight("{site}")
display(pd.DataFrame([asdict(report_{site})]))
if report_{site}.errors:
    print("ERREURS:", *report_{site}.errors, sep="\n- ")
if report_{site}.missing_year_months:
    print("MOIS S2 MANQUANTS:", ", ".join(report_{site}.missing_year_months))
print("READY =", report_{site}.ready)
"""
        ),
    ]

    if historical:
        cells.extend(
            [
                markdown(
                    f"""
## {chapter}.2 — Script B4 Ifran conservé, entraînement désactivé

`RUN_TRAIN_IFRAN=False` est volontaire : le checkpoint historique est réutilisé. La cellule écrit tout de même
la commande exacte et contrôle que la suppression des indices PALSAR 12,13 produit bien l'ordre C15.
"""
                ),
                code(
                    """
assert RUN_TRAIN_IFRAN is False, "Protection Ifran: RUN_TRAIN_IFRAN doit rester False."
if report_ifran.ready:
    checkpoint_ifran = run_train("ifran", execute=RUN_TRAIN_IFRAN)
    print("Checkpoint Ifran réutilisé:", checkpoint_ifran)
else:
    print("Ifran non disponible; voir les erreurs du préflight.")
"""
                ),
                markdown(f"## {chapter}.3 — Audit visuel STEP05 des données Ifran"),
                code(
                    """
if report_ifran.ready:
    figures_step05_ifran = make_step05_plots("ifran")
    show_pngs(figures_step05_ifran, "Ifran — STEP05")
"""
                ),
                markdown(f"## {chapter}.4 — STEP07 et graphes finaux Ifran (sans réentraînement)"),
                code(
                    """
if report_ifran.ready:
    figures_step07_ifran = run_step07(
        "ifran", split="test", execute_if_missing=RUN_EVAL_IFRAN
    )
    if figures_step07_ifran:
        show_pngs(figures_step07_ifran, "Ifran — STEP07 TEST")
    else:
        print("Pas de graphes STEP07 existants. Laisser RUN_EVAL_IFRAN=False pour ne rien recalculer, ou le passer à True pour évaluer le checkpoint existant.")
"""
                ),
            ]
        )
        return cells

    cells.extend(
        [
            markdown(
                f"""
## {chapter}.2 — Construction du catalogue natif C15 et STEP05

La cellule est bloquée tant qu'un mois S2 monolithique requis manque. Les fichiers partiels `_Sxxxx.tif`
ne sont jamais utilisés. La normalisation robuste est ajustée sur le split train uniquement.
"""
            ),
            code(
                f"""
if report_{site}.ready and RUN_BUILD_{upper}:
    summary_{site} = materialise_catalog("{site}", overwrite=False)
    display(summary_{site})
    if not (site.catalog_root / "sample_catalog_step05.csv").exists():
        run_step05("{site}")
elif not report_{site}.ready:
    print("BUILD BLOQUÉ: téléchargement S2 incomplet ou préflight invalide.")
else:
    print("Construction désactivée par RUN_BUILD_{upper}=False.")
"""
            ),
            markdown(f"## {chapter}.3 — Graphes d'audit des données STEP05"),
            code(
                f"""
if (site.catalog_root / "sample_catalog_step05.csv").exists():
    figures_step05_{site} = make_step05_plots("{site}")
    show_pngs(figures_step05_{site}, "{label} — STEP05")
else:
    print("STEP05 non disponible: exécuter d'abord la construction C15.")
"""
            ),
            markdown(
                f"""
## {chapter}.4 — Entraînement B4 Phase 1, entrée C15

Recette Ifran B4 : HyTec `base_ch=64`, dropout 0.15, Huber β=3, AdamW, seed 42,
sampler équilibré en hauteur, validation toutes les 66 étapes et patience 10 évaluations.
"""
            ),
            code(
                f"""
if report_{site}.ready and RUN_TRAIN_{upper} and (site.catalog_root / "sample_catalog_step05.csv").exists():
    checkpoint_{site} = run_train("{site}", execute=True)
    print("Checkpoint:", checkpoint_{site})
elif not report_{site}.ready:
    print("TRAIN BLOQUÉ: téléchargement S2 incomplet ou préflight invalide.")
elif not RUN_TRAIN_{upper}:
    run_train("{site}", execute=False)
else:
    print("TRAIN BLOQUÉ: catalogue STEP05 absent.")
"""
            ),
            markdown(f"## {chapter}.5 — Évaluation STEP07 et graphes à la fin du train"),
            code(
                f"""
checkpoint_exists_{site} = any((site.run_dir / "checkpoints" / name).exists() for name in ["best.ckpt", "best_any.ckpt"])
if checkpoint_exists_{site}:
    figures_step07_{site} = run_step07(
        "{site}", split="test", execute_if_missing=RUN_EVAL_{upper}
    )
    show_pngs(figures_step07_{site}, "{label} — STEP07 TEST")
else:
    print("STEP07 en attente: aucun checkpoint B4 C15 disponible.")
"""
            ),
        ]
    )
    return cells


def build_notebook() -> dict:
    cells: list[dict] = [
        markdown(
            """
# Pipeline publication B4 — C15 sans PALSAR

## Ifran · Maamoura · Agadir

Ce notebook reproduit la recette d'entraînement B4 d'Ifran avec une entrée effective strictement **C15**.
Il est organisé en trois chapitres indépendants, protège le train Ifran déjà terminé et centralise scripts,
figures, rapports, catalogues et nouveaux runs sous `C:\\Users\\Dell\\Desktop\\Publication_Inchallah`.
"""
        ),
        markdown("## Configuration générale et drapeaux d'exécution"),
        code(
            r"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

PUBLICATION_ROOT = Path(r"C:\Users\Dell\Desktop\Publication_Inchallah")
if str(PUBLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLICATION_ROOT))

from b4_c15_config import B4_C15_CHANNEL_ORDER, SITES, assert_c15_contract
from b4_c15_preflight import run_preflight
from build_b4_c15_catalog import materialise_catalog, run_step05
from run_b4_c15_train import run as run_train
from step05_visual_audit_article import make_plots as make_step05_plots
from step07_evaluate_and_plot import run as run_step07

# Ifran est déjà entraîné : ce drapeau DOIT rester False.
RUN_TRAIN_IFRAN = False
RUN_EVAL_IFRAN = False

# Les deux nouveaux sites restent automatiquement bloqués jusqu'à disponibilité complète des S2.
RUN_BUILD_MAAMOURA = True
RUN_TRAIN_MAAMOURA = True
RUN_EVAL_MAAMOURA = True

RUN_BUILD_AGADIR = True
RUN_TRAIN_AGADIR = True
RUN_EVAL_AGADIR = True

def show_pngs(paths, title):
    paths = [Path(path) for path in paths if Path(path).exists()]
    print(f"{title}: {len(paths)} figure(s)")
    for path in paths:
        display(Image(filename=str(path)))

assert len(B4_C15_CHANNEL_ORDER) == 15
assert_c15_contract(B4_C15_CHANNEL_ORDER)
print("Contrat actif: B4 C15 sans PALSAR")
"""
        ),
    ]
    cells.extend(forest_cells("ifran", 1, "Ifran", historical=True))
    cells.extend(forest_cells("maamoura", 2, "Maamoura"))
    cells.extend(forest_cells("agadir", 3, "Agadir"))
    cells.extend(
        [
            markdown("# Synthèse finale — disponibilité et artefacts"),
            code(
                """
rows = []
for key, cfg in SITES.items():
    report = run_preflight(key)
    rows.append({
        "forêt": cfg.label,
        "préflight_ready": report.ready,
        "mois_S2_manquants": len(report.missing_year_months),
        "catalogue_STEP05": (cfg.catalog_root / "sample_catalog_step05.csv").exists(),
        "checkpoint_best": (cfg.run_dir / "checkpoints" / "best.ckpt").exists(),
        "figures_STEP05": len(list((cfg.figures_root / "step05").glob("*.png"))),
        "figures_STEP07": len(list((cfg.figures_root / "step07").rglob("*.png"))),
        "catalogue": str(cfg.catalog_root),
        "run": str(cfg.run_dir),
    })
display(pd.DataFrame(rows))
"""
            ),
        ]
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.10 - CHM (.venv310)",
                "language": "python",
                "name": "chm-venv310",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("B4_C15_Ifran_Maamoura_Agadir.ipynb"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
