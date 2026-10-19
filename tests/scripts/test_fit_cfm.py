"""Smoke test: run docs/scripts/fit_cfm.py against the dummy adata fixture."""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
from anndata import AnnData
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "docs" / "scripts" / "fit_cfm.py"
CONFIG_DIR = (SCRIPT_PATH.parent / ".." / "configs").resolve()

if not SCRIPT_PATH.is_file():
    raise RuntimeError(f"training script not found at {SCRIPT_PATH}")
if not (CONFIG_DIR / "fit_cfm.yaml").is_file():
    raise RuntimeError(f"fit_cfm.yaml not found at {CONFIG_DIR / 'fit_cfm.yaml'}.")


def _load_fit_cfm() -> ModuleType:
    if "fit_cfm_script" in sys.modules:
        return sys.modules["fit_cfm_script"]
    spec = importlib.util.spec_from_file_location("fit_cfm_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fit_cfm_script"] = module
    spec.loader.exec_module(module)
    return module


fit_cfm = _load_fit_cfm()
fit_cfm_main = fit_cfm.main


def _render(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value in ("", "null", "true", "false") or any(c in value for c in ":[]{},.'\" \t"):
            return "'" + value.replace("'", "''") + "'"
        return value
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_render(k)}: {_render(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render(v) for v in value) + "]"
    raise TypeError(f"cannot render {type(value).__name__} as Hydra override")


def overrides(section: str, **fields) -> list[str]:
    out: list[str] = []
    for key, value in fields.items():
        path = f"{section}.{key}"
        if isinstance(value, (dict, list, tuple)):
            out.append(f"~{path}")
            out.append(f"+{path}={_render(value)}")
        else:
            out.append(f"{path}={_render(value)}")
    return out


# --------------------------- section variants ------------------------------

_OVERRIDES_SPLITTER: list[tuple[str, list[str]]] = [
    (
        "base",
        overrides(
            "splitter",
            group_keys=["source_split"],
            always_train_keys=[],
            control_key="drugA",
            control_value="control",
        ),
    ),
]


_DM_SHARED = {
    "conditions": {"drug": ["drugA"]},
    "conditions_reps": {"drug": "drug"},
    "conditions_covariates": [],
    "control_values_dict": {"drug": "control"},
    "groups": ["source_split"],
    "groups_encoding": {"source_split": "label"},
    "groups_reps": {},
    "matched_pairs": [],
    "rep_lin": None,
    "source_rep_quad": None,
    "target_rep_quad": None,
    "use_linear": True,
    "target_categorical_covs_dict": {},
    "target_continuous_covs": [],
}

_OVERRIDES_DM: list[tuple[str, list[str]]] = [
    ("base", overrides("datamanager", sample_rep="X_repr", **_DM_SHARED)),
    ("sample_rep_x_src", overrides("datamanager", sample_rep="X_src", **_DM_SHARED)),
]


_OVERRIDES_MODULE: list[tuple[str, list[str]]] = [
    ("base", []),
    (
        "minimal",
        overrides(
            "module",
            num_time_features=8,
            state_encoder_output_dim=16,
            time_encoder_output_dim=16,
            source_encoder_output_dim=16,
            condition_encoder_output_dim=8,
            condition_encoder_pooling_proj_dim=8,
        ),
    ),
]


_OVERRIDES_INFERENCE: list[tuple[str, list[str]]] = [
    ("base", []),
    (
        "fast",
        overrides(
            "inference",
            n_samples=16,
            n_steps=5,
            solver_kwargs={"method": "euler"},
        ),
    ),
]


_OVERRIDES_DMOD: list[tuple[str, list[str]]] = [
    (
        "fast",
        overrides(
            "datamodule",
            batch_size=32,
            n_train_steps=100,
            loader_kwargs={},
        ),
    ),
]


def _lookup(variants, name):
    for variant_name, variant_lines in variants:
        if variant_name == name:
            return variant_lines
    raise KeyError(f"unknown variant {name!r}; available: {[n for n, _ in variants]}")


def combine(
    *,
    splitter: str = "base",
    dm: str = "base",
    dmod: str = "fast",
    module: str = "minimal",
    inference: str = "fast",
    **extras: list[str],
) -> list[str]:
    out: list[str] = []
    out += _lookup(_OVERRIDES_SPLITTER, splitter)
    out += _lookup(_OVERRIDES_DM, dm)
    out += _lookup(_OVERRIDES_DMOD, dmod)
    out += _lookup(_OVERRIDES_MODULE, module)
    out += _lookup(_OVERRIDES_INFERENCE, inference)
    for extra in extras.values():
        out += extra
    return out


# --------------------------- composition -----------------------------------


def _compose(*, target_adata_path, tmp_path, overrides, config_name="fit_cfm"):
    ephemeral = [
        f"paths.target_adata={target_adata_path}",
        "paths.control_adata=null",
        f"save.run_dir={tmp_path / 'run'}",
        f"callbacks.model_checkpoint.dirpath={tmp_path / 'ckpt'}",
        f"loggers.wandb.save_dir={tmp_path / 'wandb'}",
        f"loggers.csv.save_dir={tmp_path / 'csv'}",
        f"trainer.default_root_dir={tmp_path / 'logs'}",
        "trainer.fast_dev_run=true",
        "trainer.enable_progress_bar=false",
        "loggers.wandb.offline=true",
        "run.allow_overwrite=true",
        "hydra.job.chdir=false",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides + ephemeral)


# --------------------------- fixtures --------------------------------------


@pytest.fixture
def target_adata_path(tmp_path: Path, adata: AnnData) -> Path:
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    path = tmp_path / "target.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.fixture
def compose_fit_cfm(tmp_path: Path, target_adata_path: Path) -> DictConfig:
    return _compose(
        target_adata_path=target_adata_path,
        tmp_path=tmp_path,
        overrides=combine(),
    )


# --------------------------- tests -----------------------------------------


def test_script_loads() -> None:
    assert hasattr(fit_cfm, "main")
    assert callable(fit_cfm_main.__wrapped__)
    assert fit_cfm.logger.name == "fit_cfm_script"


@pytest.mark.slow
@pytest.mark.parametrize(
    "overrides_list",
    [
        pytest.param(combine(), id="base"),
        pytest.param(combine(dm="sample_rep_x_src"), id="dm-sample-rep-x-src"),
        pytest.param(
            combine(coupling=["coupling.method=sinkhorn"]),
            id="coupling-sinkhorn",
        ),
        pytest.param(
            combine(probability_path=["probability_path.name=schrodinger_bridge"]),
            id="schrodinger-bridge",
        ),
    ],
)
def test_fit_cfm_runs(
    overrides_list: list[str],
    tmp_path: Path,
    target_adata_path: Path,
) -> None:
    cfg = _compose(
        target_adata_path=target_adata_path,
        tmp_path=tmp_path,
        overrides=overrides_list,
    )

    try:
        score = fit_cfm_main.__wrapped__(cfg)
    except Exception:
        fit_cfm.logger.exception("fit_cfm.main() raised")
        raise

    assert isinstance(score, float)
    assert not math.isnan(score)

    run_dir = tmp_path / "run"
    assert (run_dir / "specs.json").is_file()
    assert (run_dir / "weights.pt").is_file()
