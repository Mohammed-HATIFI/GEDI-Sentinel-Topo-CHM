from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PUBLICATION_ROOT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
PYTHON = Path(r"C:\Users\Dell\Desktop\Article_Maroc\Env_Workspace_Maamoura\.venv310\Scripts\python.exe")
VENDOR_ROOT = PUBLICATION_ROOT / "Source" / "Training" / "vendor_b4_trainer"
TRAIN_SCRIPT = VENDOR_ROOT / "06_train_growthloss_catalog.py"
STEP05_SCRIPT = VENDOR_ROOT / "preprocess_eval_scripts" / "step05_build_catalog_npy.py"
STEP07_SCRIPT = VENDOR_ROOT / "preprocess_eval_scripts" / "step07_eval_original_coords_minmax.py"

RAW_S2_ORDER = (
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
)
B4_S2_ORDER = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A")
B4_C15_CHANNEL_ORDER = (
    "S2_LEAFON_B02", "S2_LEAFON_B03", "S2_LEAFON_B04", "S2_LEAFON_B08",
    "S2_LEAFON_B05", "S2_LEAFON_B06", "S2_LEAFON_B07", "S2_LEAFON_B8A",
    "S1_ASC_VV", "S1_ASC_VH", "S1_DESC_VV", "S1_DESC_VH",
    "AOI_MASK", "DEM", "SLOPE",
)
SOURCE_C11_CHANNEL_ORDER = (
    "S2_LEAFON_B02", "S2_LEAFON_B03", "S2_LEAFON_B04", "S2_LEAFON_B08",
    "S1_ASC_VV", "S1_ASC_VH", "S1_DESC_VV", "S1_DESC_VH",
    "PALSAR_HH", "PALSAR_HV", "AOI_MASK",
)


@dataclass(frozen=True)
class SiteConfig:
    key: str
    label: str
    source_npz_experiment: Path | None
    s2_dir: Path | None
    dem: Path | None
    slope: Path | None
    catalog_root: Path
    runs_root: Path
    run_name: str
    train_min: float
    train_max: float
    eval_min: float
    eval_max: float
    batch_size: int
    native_c15: bool = True
    historical_drop_channels: tuple[int, ...] = ()
    existing_run_dir: Path | None = None
    existing_experiment_root: Path | None = None
    figures_root_override: Path | None = None
    reports_root_override: Path | None = None

    @property
    def figures_root(self) -> Path:
        if self.figures_root_override is not None:
            return self.figures_root_override
        return PUBLICATION_ROOT / "Results" / self.key / "Figures"

    @property
    def reports_root(self) -> Path:
        if self.reports_root_override is not None:
            return self.reports_root_override
        return PUBLICATION_ROOT / "Documentation" / "Run_Reports" / self.key

    @property
    def run_dir(self) -> Path:
        if self.existing_run_dir is not None and self.run_name.endswith("_FROZEN_REFERENCE"):
            return self.existing_run_dir
        return self.runs_root / self.catalog_root.name / self.run_name


SITES: dict[str, SiteConfig] = {
    "ifran": SiteConfig(
        key="ifran", label="Ifran", source_npz_experiment=None,
        s2_dir=Path(r"E:\CHM\Ifran_6\DATA\S2\S2_MONTHLY_IFRAN_CLEAN_V1"),
        dem=Path(r"E:\CHM\Ifran_6\DATA\DEM\DEM_SRTM_AOI_EPSG32629_30m.tif"),
        slope=Path(r"E:\CHM\Ifran_6\DATA\DEM\SLOPE_QGIS.tif"),
        catalog_root=PUBLICATION_ROOT / "Data" / "Dense" / "Ifran" / "Catalogs" / "final_catalog_C15_NATIVE",
        runs_root=PUBLICATION_ROOT / "Runs" / "Dense" / "Ifran",
        run_name="IFRAN_B4_C15_FACTORIAL_H3_VAL2_45_TEST0_45_B8_P15_SEED42_REPRODUCTION",
        train_min=0.0, train_max=45.0, eval_min=2.0, eval_max=45.0, batch_size=8,
        native_c15=True, historical_drop_channels=(),
        existing_experiment_root=PUBLICATION_ROOT / "Data" / "Dense" / "Ifran" / "Catalogs" / "final_catalog_C15_NATIVE",
        existing_run_dir=PUBLICATION_ROOT / "Models" / "Dense" / "Ifran",
    ),
    "maamoura": SiteConfig(
        key="maamoura", label="Maamoura",
        source_npz_experiment=Path(
            r"C:\Users\Dell\Desktop\Article_Maroc\Publication\Shard_NPZ"
            r"\exp_maamoura_gediids_uniqueval_randomstrat_afterqc_strictqc_s2aoi90_hytec_75_15_10_PATCH512_STRIDE512_S1ASC_DESC_C11_AOIMASK_MAXHEIGHT20M_TRAIN0_VALTEST0_PALSAR_v12"
        ),
        s2_dir=Path(r"E:\CHM\Maamoura\Data\S2_12_Bands"),
        dem=Path(r"E:\CHM\Maamoura\Data\DEM\DEM_SRTM_AOI_EPSG32629_30m.tif"),
        slope=Path(r"E:\CHM\Maamoura\Data\DEM\Slope_Maamoura_Qgis.tif"),
        catalog_root=PUBLICATION_ROOT / "Data" / "Low_Sparsity" / "Maamoura" / "Catalogs" / "final_catalog",
        runs_root=PUBLICATION_ROOT / "Runs" / "Low_Sparsity" / "Maamoura",
        run_name="MAAMOURA_H1_VAL2_B4_C15_B8_P15_SEED42_FIXED_SPLIT_REPRODUCTION",
        train_min=0.0, train_max=20.0, eval_min=2.0, eval_max=20.0, batch_size=8,
    ),
    "agadir": SiteConfig(
        key="agadir", label="Agadir",
        source_npz_experiment=Path(
            r"E:\CHM\Agadir\Output\Publication\Shard_NPZ"
            r"\exp_agadir_oldspatial70_15_15_tile512_patch512_balanced_PATCH512_STRIDE512_C11_PALSAR_AOIMASK_RH95_FULLDOMAIN0_20_PALSAR_NOPAULS_COMMONCORE_POINTLEVELGEDI_v23"
        ),
        s2_dir=Path(r"E:\CHM\Agadir\DATA\S2_12_Bands"),
        dem=Path(r"E:\CHM\Agadir\DATA\DEM\DEM_SRTM_AOI_EPSG32629_30m.tif"),
        slope=Path(r"E:\CHM\Agadir\DATA\DEM\SLOPE_QGIS.tif"),
        catalog_root=PUBLICATION_ROOT / "Data" / "Sparse" / "Agadir" / "Catalogs" / "final_catalog",
        runs_root=PUBLICATION_ROOT / "Runs" / "Sparse" / "Agadir",
        run_name="SPARSE_AGADIR_B4_C15_H3_VAL2_B8_P15_SEED42_REPRODUCTION",
        train_min=0.0, train_max=20.0, eval_min=2.0, eval_max=20.0, batch_size=8,
    ),
}


def ensure_publication_dirs() -> None:
    for site in SITES.values():
        site.figures_root.mkdir(parents=True, exist_ok=True)
        site.reports_root.mkdir(parents=True, exist_ok=True)
        site.runs_root.mkdir(parents=True, exist_ok=True)


def assert_c15_contract(channels: list[str] | tuple[str, ...]) -> None:
    values = tuple(str(value) for value in channels)
    if values != B4_C15_CHANNEL_ORDER:
        raise RuntimeError(f"B4 C15 channel contract mismatch. Expected={B4_C15_CHANNEL_ORDER}; found={values}")
    if any("PALSAR" in name.upper() for name in values):
        raise RuntimeError("PALSAR must not enter the effective B4 C15 model input.")
