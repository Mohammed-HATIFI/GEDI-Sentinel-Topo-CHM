from __future__ import annotations

"""Evaluate and rank every saved B4 checkpoint for Ifran, Maamoura and Agadir.

The script evaluates each checkpoint once on the widest TEST domain, caches the
occurrence predictions in an isolated staging directory, then recomputes
unique-nearest metrics for every requested RH95 domain. Existing run STEP07
folders are never overwritten.

KGE is the original Gupta et al. (2009) formulation:
    1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
where r is Pearson correlation, alpha is std(pred)/std(obs), and beta is
mean(pred)/mean(obs).
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PUBLICATION_ROOT = Path(r"C:\Users\Dell\Desktop\Publication_Inchallah")
CLARCK_ROOT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
PYTHON = Path(
    r"C:\Users\Dell\Desktop\Article_Maroc\Env_Workspace_Maamoura"
    r"\.venv310\Scripts\python.exe"
)
EVALUATOR = (
    PUBLICATION_ROOT
    / "vendor_b4_trainer"
    / "preprocess_eval_scripts"
    / "step07_eval_original_coords_minmax.py"
)
OUTPUT_ROOT = CLARCK_ROOT / "B4_C15_ALL_RUNS_KGE_RANKING"

# These fixed-split Maamoura ablations were completed after the first global
# ranking.  Keep them explicit: their run names are generated dynamically in
# the notebook and may therefore be absent from the notebook's literal text.
REQUIRED_NEW_MAAMOURA_RUNS = {
    "MAAMOURA_H1_VAL2_B4_C15_B8_SEED42_FIXED_SPLIT",
    "MAAMOURA_H1_VAL2P5_B4_C15_B8_SEED42_FIXED_SPLIT",
    "MAAMOURA_H3_VAL2P5_B4_C15_B8_SEED42_FIXED_SPLIT",
}

REQUIRED_FACTORIAL_IFRAN_RUNS = {
    "IFRAN_B4_C15_FACTORIAL_H1_VAL2_45_TEST0_45_B8_P15_SEED42",
    "IFRAN_B4_C15_FACTORIAL_H1_VAL2P5_45_TEST0_45_B8_P15_SEED42",
    "IFRAN_B4_C15_FACTORIAL_H3_VAL2_45_TEST0_45_B8_P15_SEED42",
}

REQUIRED_EXPLICIT_RUNS = (
    REQUIRED_NEW_MAAMOURA_RUNS
    | REQUIRED_FACTORIAL_IFRAN_RUNS
)

# The notebooks are the provenance authority requested for this audit.
NOTEBOOK_SOURCES = {
    "Three ecosystems publication": (
        CLARCK_ROOT / "B4_C15_Three_Ecosystems_Publication.ipynb"
    ),
    "Ifran Maamoura Agadir ablations": (
        PUBLICATION_ROOT / "B4_C15_Ifran_Maamoura_Agadir.ipynb"
    ),
    "Ifran Echosat Disturbance ablations": (
        PUBLICATION_ROOT / "Ablation_Seuil_Echosat_Disturbance.ipynb"
    ),
}

# Some Ifran runs referenced by the third notebook are outside Publication roots.
SEARCH_ROOTS = [
    CLARCK_ROOT,
    PUBLICATION_ROOT,
    Path(r"C:\Users\Dell\Desktop\Safi_3\Last_Ablation_Output\Runs_NoPalsar_Cumulative_B2_to_B5"),
    Path(r"C:\Users\Dell\Documents\CHM_Maroc\ECHOSAT_AntiShrink_GrowthLoss_v5"),
]

CHECKPOINT_NAMES = (
    "best.ckpt",
    "best_any.ckpt",
    "best_compromise.ckpt",
    "best_slope.ckpt",
    "best_r2.ckpt",
)

# TEST is the publication comparison. Add "val" to audit model selection too.
EVALUATION_SPLITS = ("test",)
EXECUTE_MISSING_EVALUATIONS = True
EVALUATION_BATCH_SIZE = 4
MAX_RUNS: int | None = None
REQUIRE_NOTEBOOK_REFERENCE = True

SITE_DOMAINS = {
    "ifran": (
        (0.0, 45.0), (2.0, 45.0), (2.5, 45.0),
        (0.0, 40.0), (2.0, 40.0), (2.5, 40.0),
        (0.0, 20.0), (2.0, 20.0), (2.5, 20.0),
    ),
    "maamoura": ((0.0, 20.0), (2.0, 20.0), (2.5, 20.0)),
    "agadir": ((0.0, 20.0), (2.0, 20.0), (2.5, 20.0)),
}

SITE_LABELS = {
    "ifran": ("Dense forest", "Ifran"),
    "maamoura": ("Low-sparsity forest", "Maamoura"),
    "agadir": ("Sparse forest", "Agadir"),
}


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _site_from_run(run_name: str, run_dir: Path) -> str | None:
    """Classify from the actual forest token, not protocol names in suffixes."""
    name = run_name.lower()
    explicit_prefixes = {
        "maamoura": ("maamoura", "lowsp", "lowsparsity"),
        "agadir": ("agadir", "sparse_agadir"),
        "ifran": ("ifran", "dense_ifran"),
    }
    for site, prefixes in explicit_prefixes.items():
        if name.startswith(prefixes):
            return site

    components = [part.lower() for part in run_dir.parts]
    for site in ("maamoura", "agadir", "ifran"):
        if any(
            component == site
            or component.startswith(site + "_")
            or component.endswith("_" + site)
            for component in components
        ):
            return site
    return None


def _domain_label(lower: float, upper: float) -> str:
    return f"{lower:g}-{upper:g} m"


def _run_token(run_dir: Path) -> str:
    digest = hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:12]
    return f"{run_dir.name[:80]}__{digest}"


def _load_run_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload.get("args", payload)
    if not isinstance(args, dict):
        raise TypeError(f"Invalid args in {path}")
    return args


def _notebook_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for label, path in NOTEBOOK_SOURCES.items():
        if not path.is_file():
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            texts[label] = "\n".join(
                "".join(cell.get("source", []))
                + "\n"
                + "\n".join(
                    str(output.get("text", ""))
                    for output in cell.get("outputs", [])
                    if isinstance(output, dict)
                )
                for cell in notebook.get("cells", [])
            ).lower()
        except Exception as exc:
            print(f"[WARN] Cannot parse provenance notebook {path}: {exc!r}")
    return texts


def _run_notebook_provenance(
    run_name: str,
    run_dir: Path,
    notebook_texts: dict[str, str],
) -> tuple[str, str]:
    needles = (run_name.lower(), str(run_dir).lower())
    labels = [
        label
        for label, text in notebook_texts.items()
        if any(needle and needle in text for needle in needles)
    ]
    if not labels:
        return "", "not_referenced"
    specialized = (
        labels == ["Ifran Echosat Disturbance ablations"]
        or (
            "Ifran Echosat Disturbance ablations" in labels
            and "Three ecosystems publication" not in labels
            and "Ifran Maamoura Agadir ablations" not in labels
        )
    )
    tier = "ifran_specialized_ablation" if specialized else "main_three_ecosystem_protocol"
    return " | ".join(labels), tier


def discover_runs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    notebook_texts = _notebook_texts()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for config_path in root.rglob("run_config.json"):
            if config_path.parent.name != "artifacts":
                continue
            run_dir = config_path.parent.parent
            resolved = str(run_dir.resolve()).lower()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                args = _load_run_config(config_path)
                experiment_root = Path(str(args["experiment_root"]))
                run_name = str(args.get("run_name", run_dir.name))
                site = _site_from_run(run_name, run_dir)
                if site is None:
                    rejected.append({"run_dir": str(run_dir), "reason": "site_not_identified"})
                    continue
                source_notebooks, comparison_tier = _run_notebook_provenance(
                    run_name, run_dir, notebook_texts
                )
                explicitly_required = run_name in REQUIRED_EXPLICIT_RUNS
                if REQUIRE_NOTEBOOK_REFERENCE and not source_notebooks and not explicitly_required:
                    rejected.append({
                        "run_dir": str(run_dir),
                        "reason": "not_referenced_by_requested_notebooks",
                    })
                    continue
                if explicitly_required:
                    if run_name in REQUIRED_NEW_MAAMOURA_RUNS:
                        explicit_registry = (
                            "Explicit fixed-split Maamoura ablation registry"
                        )
                    else:
                        explicit_registry = (
                            "Explicit factorial Ifran ablation registry"
                        )
                    source_notebooks = source_notebooks or explicit_registry
                    comparison_tier = "main_three_ecosystem_protocol"
                checkpoints = [
                    run_dir / "checkpoints" / name
                    for name in CHECKPOINT_NAMES
                    if (run_dir / "checkpoints" / name).is_file()
                ]
                if not checkpoints:
                    rejected.append({"run_dir": str(run_dir), "reason": "no_requested_checkpoint"})
                    continue
                if not experiment_root.exists():
                    rejected.append({
                        "run_dir": str(run_dir),
                        "reason": "experiment_root_missing",
                        "experiment_root": str(experiment_root),
                    })
                    continue
                runs.append({
                    "site": site,
                    "run_dir": run_dir,
                    "run_name": run_name,
                    "experiment_root": experiment_root,
                    "args": args,
                    "checkpoints": checkpoints,
                    "config_path": config_path,
                    "source_notebooks": source_notebooks,
                    "comparison_tier": comparison_tier,
                })
            except Exception as exc:
                rejected.append({"run_dir": str(run_dir), "reason": f"config_error: {exc}"})
    runs.sort(key=lambda item: (item["site"], item["run_name"], str(item["run_dir"])))
    discovered_names = {item["run_name"] for item in runs}
    missing_required = sorted(REQUIRED_EXPLICIT_RUNS - discovered_names)
    if missing_required:
        raise RuntimeError(
            "Required completed ablations were not discovered: "
            + ", ".join(missing_required)
        )
    if MAX_RUNS is not None:
        runs = runs[: int(MAX_RUNS)]
    return runs, rejected


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prediction_cache(run: dict[str, Any], checkpoint: Path, split: str) -> Path:
    return (
        OUTPUT_ROOT
        / "Prediction_Cache"
        / run["site"]
        / _run_token(run["run_dir"])
        / checkpoint.stem
        / f"{split}_predictions_full_domain.csv.gz"
    )


def evaluate_checkpoint(run: dict[str, Any], checkpoint: Path, split: str) -> Path:
    cache_path = _prediction_cache(run, checkpoint, split)
    if cache_path.is_file():
        print("[CACHE]", run["run_name"], checkpoint.name, split)
        return cache_path
    if not EXECUTE_MISSING_EVALUATIONS:
        raise FileNotFoundError(f"Missing ranking cache: {cache_path}")

    site = run["site"]
    upper = max(domain[1] for domain in SITE_DOMAINS[site])
    run_token = _run_token(run["run_dir"])
    staging_runs_root = OUTPUT_ROOT / "_Evaluator_Stage" / site / run_token
    # A short stage alias avoids Windows MAX_PATH failures for deeply nested runs.
    stage_run_name = "run_" + hashlib.sha1(
        (str(run["run_dir"]) + checkpoint.name).encode("utf-8")
    ).hexdigest()[:16]
    staged_run = staging_runs_root / run["experiment_root"].name / stage_run_name
    staged_checkpoint = staged_run / "checkpoints" / checkpoint.name
    _link_or_copy(checkpoint, staged_checkpoint)

    args = run["args"]
    command = [
        str(PYTHON if PYTHON.is_file() else Path(sys.executable)),
        str(EVALUATOR),
        "--experiment-root", str(run["experiment_root"]),
        "--runs-root", str(staging_runs_root),
        "--run-name", stage_run_name,
        "--split", split,
        "--checkpoint-name", checkpoint.name,
        "--batch-size", str(EVALUATION_BATCH_SIZE),
        "--max-aux-k", str(int(args.get("max_aux_k", 2048))),
        "--min-height", "0",
        "--max-height", f"{upper:g}",
        "--bins", "0,2,2.5,5,8,10,15,20,30,40,45",
        "--no-clip-predictions",
        "--no-plots",
    ]
    drop_channels = args.get("drop_channels")
    if drop_channels not in (None, "", []):
        if isinstance(drop_channels, (list, tuple)):
            drop_channels = ",".join(str(value) for value in drop_channels)
        command.extend(["--drop-channels", str(drop_channels)])

    print("[EVALUATE]", run["run_name"], checkpoint.name, split)
    print(subprocess.list2cmdline(command))
    subprocess.run(command, check=True)
    generated_dir = staged_run / "step07_original_coords_minmax"
    generated_predictions = generated_dir / f"{split}_predictions_original_coords.csv.gz"
    generated_metrics = generated_dir / f"{split}_metrics_original_coords.json"
    if not generated_predictions.is_file():
        raise FileNotFoundError(generated_predictions)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_predictions, cache_path)
    if generated_metrics.is_file():
        shutil.copy2(generated_metrics, cache_path.with_suffix("").with_suffix(".json"))
    provenance = {
        "run_dir": str(run["run_dir"]),
        "original_run_name": run["run_name"],
        "staged_run_name": stage_run_name,
        "run_config": str(run["config_path"]),
        "experiment_root": str(run["experiment_root"]),
        "checkpoint": str(checkpoint),
        "split": split,
        "command": command,
        "prediction_clipping": False,
        "evaluation_coordinates": "original_GEDI_coordinates",
    }
    cache_path.with_name(cache_path.name + ".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return cache_path


def unique_nearest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "aux_shot_uid", "abs_temporal_delta_days", "rh95",
        "prediction_original_coords",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Prediction file is missing columns: {sorted(missing)}")
    work = frame.copy()
    work["_delta"] = pd.to_numeric(
        work["abs_temporal_delta_days"], errors="coerce"
    ).fillna(np.inf)
    work["_row"] = np.arange(len(work))
    return (
        work.sort_values(["aux_shot_uid", "_delta", "_row"])
        .drop_duplicates("aux_shot_uid", keep="first")
        .copy()
    )


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    y = observed[valid]
    p = predicted[valid]
    n = int(len(y))
    result: dict[str, Any] = {"n": n}
    if n == 0:
        for name in (
            "mae", "rmse", "r2", "bias", "slope", "corr", "std_ratio",
            "observed_mean", "predicted_mean", "kge_r", "kge_alpha",
            "kge_beta", "kge", "pred_zero_pct",
        ):
            result[name] = math.nan
        return result

    error = p - y
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    y_mean = float(np.mean(y))
    p_mean = float(np.mean(p))
    y_std = float(np.std(y, ddof=0))
    p_std = float(np.std(p, ddof=0))
    denominator = float(np.sum((y - y_mean) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else math.nan
    if n >= 2 and y_std > 0 and p_std > 0:
        corr = float(np.corrcoef(y, p)[0, 1])
        slope = float(np.polyfit(y, p, 1)[0])
    else:
        corr = math.nan
        slope = math.nan
    alpha = float(p_std / y_std) if y_std > 0 else math.nan
    beta = float(p_mean / y_mean) if y_mean != 0 else math.nan
    if all(math.isfinite(value) for value in (corr, alpha, beta)):
        kge = float(1.0 - math.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = math.nan
    result.update({
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "slope": slope,
        "corr": corr,
        "std_ratio": alpha,
        "observed_mean": y_mean,
        "predicted_mean": p_mean,
        "kge_r": corr,
        "kge_alpha": alpha,
        "kge_beta": beta,
        "kge": kge,
        "pred_zero_pct": float(100.0 * np.mean(p <= 0.0)),
    })
    return result


def run_metadata(run: dict[str, Any]) -> dict[str, Any]:
    args = run["args"]
    return {
        "source_notebooks": run["source_notebooks"],
        "comparison_tier": run["comparison_tier"],
        "huber_delta": _finite(args.get("huber_beta")),
        "regression_loss": args.get("regression_loss"),
        "model_type": args.get("model_type"),
        "beta_ordinal": _finite(args.get("beta_ord")),
        "gamma_classification": _finite(args.get("gamma_cls")),
        "training_val_min": _finite(args.get("eval_primary_min_height")),
        "training_val_max": _finite(args.get("eval_primary_max_height")),
        "batch_size": args.get("batch_size"),
        "patience_evals": args.get("step1_patience_evals"),
        "seed": args.get("seed"),
        "monitor": args.get("phase1_monitor", args.get("monitor")),
    }


def rank_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runs, rejected = discover_runs()
    pd.DataFrame(rejected).to_csv(OUTPUT_ROOT / "rejected_runs.csv", index=False)
    print(f"[DISCOVERY] eligible runs={len(runs)} | rejected={len(rejected)}")
    inventory_rows = []
    metric_rows = []
    error_rows = []

    for run in runs:
        ecosystem, forest = SITE_LABELS[run["site"]]
        metadata = run_metadata(run)
        for checkpoint in run["checkpoints"]:
            inventory_rows.append({
                "ecosystem": ecosystem,
                "forest": forest,
                "run_name": run["run_name"],
                "run_dir": str(run["run_dir"]),
                "experiment_root": str(run["experiment_root"]),
                "checkpoint": checkpoint.name,
                "checkpoint_path": str(checkpoint),
                "checkpoint_size_mb": checkpoint.stat().st_size / (1024 ** 2),
                **metadata,
            })
            for split in EVALUATION_SPLITS:
                try:
                    prediction_path = evaluate_checkpoint(run, checkpoint, split)
                    occurrence_frame = pd.read_csv(prediction_path)
                    nearest = unique_nearest(occurrence_frame)
                    for lower, upper in SITE_DOMAINS[run["site"]]:
                        domain = nearest.loc[
                            pd.to_numeric(nearest["rh95"], errors="coerce").between(lower, upper)
                        ]
                        metrics = calculate_metrics(
                            domain["rh95"].to_numpy(),
                            domain["prediction_original_coords"].to_numpy(),
                        )
                        metric_rows.append({
                            "ecosystem": ecosystem,
                            "forest": forest,
                            "split": split.upper(),
                            "evaluation_unit": "unique-nearest GEDI",
                            "domain_m": _domain_label(lower, upper),
                            "domain_min_m": lower,
                            "domain_max_m": upper,
                            "run_name": run["run_name"],
                            "run_dir": str(run["run_dir"]),
                            "checkpoint": checkpoint.name,
                            "checkpoint_path": str(checkpoint),
                            "prediction_cache": str(prediction_path),
                            **metadata,
                            **metrics,
                        })
                except Exception as exc:
                    error_rows.append({
                        "forest": forest,
                        "run_name": run["run_name"],
                        "run_dir": str(run["run_dir"]),
                        "checkpoint": checkpoint.name,
                        "split": split,
                        "error": repr(exc),
                    })
                    print("[ERROR]", run["run_name"], checkpoint.name, split, repr(exc))

    inventory = pd.DataFrame(inventory_rows)
    ranking = pd.DataFrame(metric_rows)
    errors = pd.DataFrame(error_rows)
    inventory.to_csv(OUTPUT_ROOT / "checkpoint_inventory.csv", index=False)
    errors.to_csv(OUTPUT_ROOT / "evaluation_errors.csv", index=False)
    if ranking.empty:
        raise RuntimeError(
            "No metrics were produced. Inspect evaluation_errors.csv or set "
            "EXECUTE_MISSING_EVALUATIONS=True."
        )

    group = ["forest", "split", "domain_min_m", "domain_max_m"]
    ranking["rank_kge"] = ranking.groupby(group)["kge"].rank(
        method="min", ascending=False, na_option="bottom"
    ).astype("Int64")
    ranking["rank_mae"] = ranking.groupby(group)["mae"].rank(
        method="min", ascending=True, na_option="bottom"
    ).astype("Int64")
    ranking = ranking.sort_values(group + ["rank_kge", "mae", "run_name", "checkpoint"])
    ranking.to_csv(OUTPUT_ROOT / "all_checkpoint_domain_metrics.csv", index=False)

    best_per_run = (
        ranking.sort_values(group + ["run_name", "kge", "mae"], ascending=[True, True, True, True, True, False, True])
        .drop_duplicates(group + ["run_name"], keep="first")
        .sort_values(group + ["kge", "mae"], ascending=[True, True, True, True, False, True])
    )
    best_per_run.to_csv(OUTPUT_ROOT / "best_checkpoint_per_run_and_domain.csv", index=False)

    winners = (
        ranking.sort_values(group + ["kge", "mae"], ascending=[True, True, True, True, False, True])
        .groupby(group, as_index=False, sort=False)
        .head(10)
    )
    winners.to_csv(OUTPUT_ROOT / "top10_per_forest_and_domain.csv", index=False)

    # Dedicated exports for the three newly completed fixed-split Maamoura
    # ablations.  Each row is TEST unique-nearest on 0-20, 2-20 or 2.5-20 m.
    new_maamoura = ranking.loc[
        ranking["run_name"].isin(REQUIRED_NEW_MAAMOURA_RUNS)
        & ranking["forest"].eq("Maamoura")
        & ranking["split"].eq("TEST")
    ].copy()
    expected_domains = {"0-20 m", "2-20 m", "2.5-20 m"}
    observed_domains = set(new_maamoura["domain_m"].dropna().unique())
    if observed_domains != expected_domains:
        raise RuntimeError(
            "Unexpected Maamoura TEST domains: "
            f"observed={sorted(observed_domains)}, expected={sorted(expected_domains)}"
        )
    new_maamoura.to_csv(
        OUTPUT_ROOT / "maamoura_new_three_all_checkpoint_test_metrics.csv",
        index=False,
    )
    new_maamoura_best = (
        new_maamoura.sort_values(
            ["run_name", "domain_min_m", "domain_max_m", "kge", "mae"],
            ascending=[True, True, True, False, True],
        )
        .drop_duplicates(
            ["run_name", "domain_min_m", "domain_max_m"], keep="first"
        )
    )
    new_maamoura_best.to_csv(
        OUTPUT_ROOT / "maamoura_new_three_best_checkpoint_per_test_domain.csv",
        index=False,
    )

    # Publication table requested by the audit: one winning checkpoint for
    # every Forest x training Primary-VAL x TEST-domain combination.  KGE is
    # primary, followed by R2 and then MAE as deterministic tie-breakers.
    publication_domains = {
        "Ifran": {"0-45 m", "2-45 m", "2.5-45 m"},
        "Maamoura": {"0-20 m", "2-20 m", "2.5-20 m"},
        "Agadir": {"0-20 m", "2-20 m", "2.5-20 m"},
    }
    comparable = ranking.loc[
        ranking["split"].eq("TEST")
        & ranking["evaluation_unit"].eq("unique-nearest GEDI")
        & ranking["comparison_tier"].eq("main_three_ecosystem_protocol")
        & ranking["regression_loss"].astype(str).str.lower().eq("huber")
        & pd.to_numeric(ranking["beta_ordinal"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(
            ranking["gamma_classification"], errors="coerce"
        ).fillna(0).eq(0)
    ].copy()
    comparable = comparable.loc[
        comparable.apply(
            lambda row: row["domain_m"]
            in publication_domains.get(row["forest"], set()),
            axis=1,
        )
    ].drop_duplicates(
        ["checkpoint_path", "domain_min_m", "domain_max_m"]
    )
    primary_val_group = [
        "forest", "training_val_min", "training_val_max",
        "domain_min_m", "domain_max_m",
    ]
    best_by_primary_val = (
        comparable.sort_values(
            primary_val_group + ["kge", "r2", "mae"],
            ascending=[True, True, True, True, True, False, False, True],
        )
        .drop_duplicates(primary_val_group, keep="first")
        .sort_values(primary_val_group)
    )
    best_by_primary_val.to_csv(
        OUTPUT_ROOT
        / "best_model_per_forest_primary_val_and_test_domain.csv",
        index=False,
    )

    try:
        with pd.ExcelWriter(OUTPUT_ROOT / "B4_all_runs_KGE_ranking.xlsx") as writer:
            ranking.to_excel(writer, sheet_name="All checkpoint metrics", index=False)
            best_per_run.to_excel(writer, sheet_name="Best checkpoint per run", index=False)
            winners.to_excel(writer, sheet_name="Top10 by domain", index=False)
            inventory.to_excel(writer, sheet_name="Checkpoint inventory", index=False)
            errors.to_excel(writer, sheet_name="Evaluation errors", index=False)
            new_maamoura.to_excel(
                writer, sheet_name="New Maamoura all ckpts", index=False
            )
            new_maamoura_best.to_excel(
                writer, sheet_name="New Maamoura best", index=False
            )
            best_by_primary_val.to_excel(
                writer, sheet_name="Best by primary VAL", index=False
            )
    except Exception as exc:
        print("[WARN] Excel export unavailable:", repr(exc))

    print("\nTOP 10 PER FOREST / DOMAIN BY KGE")
    display_columns = [
        "forest", "split", "domain_m", "rank_kge", "run_name", "checkpoint",
        "n", "kge", "mae", "rmse", "r2", "bias", "slope", "corr",
        "std_ratio", "kge_beta", "pred_zero_pct",
    ]
    print(winners[display_columns].to_string(index=False))
    print("\nSaved ranking:", OUTPUT_ROOT)
    print("\nBEST MODEL PER FOREST / PRIMARY VAL / TEST DOMAIN")
    primary_display = [
        "forest", "training_val_min", "training_val_max", "domain_m",
        "huber_delta", "run_name", "checkpoint", "n", "kge", "mae",
        "rmse", "r2", "bias", "slope", "corr", "std_ratio",
        "pred_zero_pct",
    ]
    print(best_by_primary_val[primary_display].to_string(index=False))
    return ranking, best_per_run


if __name__ == "__main__":
    all_checkpoint_ranking, best_checkpoint_per_run = rank_all()
