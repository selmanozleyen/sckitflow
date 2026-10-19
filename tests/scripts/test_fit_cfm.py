"""Run docs/scripts/fit_cfm.py on the dummy adata, then reload what it saved."""

import importlib.util
import math
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from anndata import AnnData
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from sckitflow import load_run, predict_adata

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "docs" / "scripts" / "fit_cfm.py"
CONFIG_DIR = (SCRIPT_PATH.parent / ".." / "configs").resolve()


def _load_script():
    spec = importlib.util.spec_from_file_location("fit_cfm_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fit_cfm_script"] = module
    spec.loader.exec_module(module)
    return module


fit_cfm = sys.modules.get("fit_cfm_script") or _load_script()

# The dummy adata: drug embeddings in `uns["drug"]`, a `source_split` group, `X_repr` / `X_src` in `obsm`.
SMALL = {
    "run.splitter.group_keys": ["source_split"],
    "run.splitter.always_train_keys": [],
    "run.splitter.control_key": "drugA",
    "run.data.batch_size": 32,
    "run.data.n_train_steps": 100,
    "run.data.manager": {
        "type": "data_manager",
        "version": 1,
        "sample_rep": "X_repr",
        "conditions": {"drug": ["drugA"]},
        "conditions_reps": {"drug": "drug"},
        "control_values_dict": {"drug": "control"},
        "groups": ["source_split"],
        "groups_encoding": {"source_split": "label"},
    },
    "run.module.num_time_features": 8,
    "run.module.state_encoder_output_dim": 16,
    "run.module.time_encoder_output_dim": 16,
    "run.module.source_encoder_output_dim": 16,
    "run.module.condition_encoder_output_dim": 8,
    "run.module.condition_encoder_pooling_proj_dim": 8,
    "run.inference.n_samples": 16,
    "run.inference.n_steps": 5,
    "run.inference.solver_kwargs": {"method": "euler"},
    "trainer.fast_dev_run": True,
    "trainer.enable_progress_bar": False,
    "save.allow_overwrite": True,
}


def _config(tmp_path: Path, adata_path: Path, variant: dict):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name="fit_cfm")
    paths = {
        "paths.target_adata": str(adata_path),
        "save.run_dir": str(tmp_path / "run"),
        "callbacks.model_checkpoint.dirpath": str(tmp_path / "ckpt"),
        "loggers.wandb.save_dir": str(tmp_path / "wandb"),
        "loggers.csv.save_dir": str(tmp_path / "csv"),
        "trainer.default_root_dir": str(tmp_path / "logs"),
    }
    for key, value in {**SMALL, **paths, **variant}.items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)
    return config


@pytest.fixture
def adata_path(tmp_path: Path, adata: AnnData) -> Path:
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    path = tmp_path / "target.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.mark.slow
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param({}, id="base"),
        pytest.param({"run.data.manager.sample_rep": "X_src"}, id="sample-rep-x-src"),
        pytest.param({"run.training.coupling": {"type": "coupling.pot.sinkhorn", "version": 1}}, id="sinkhorn"),
        pytest.param(
            {"run.training.method.probability_path.type": "probability_path.schrodinger_bridge"},
            id="schrodinger-bridge",
        ),
    ],
)
def test_fit_cfm_trains_saves_and_reloads(tmp_path: Path, adata_path: Path, variant: dict) -> None:
    score = fit_cfm.main.__wrapped__(_config(tmp_path, adata_path, variant))
    assert math.isfinite(score)

    adata = ad.read_h5ad(adata_path)
    datamodule, plan = load_run(tmp_path / "run", adata)
    control = adata.obs["drugA"] == "control"
    pred = predict_adata(
        datamodule, plan.inference_method, adata[~control].copy(), control_adata=adata[control].copy(), max_per_group=4
    )
    assert pred.n_obs > 0
    assert np.isfinite(np.asarray(pred.X)).all()


def test_existing_run_fails_before_training(tmp_path: Path, adata_path: Path) -> None:
    config = _config(tmp_path, adata_path, {"save.allow_overwrite": False})
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "specs.json").write_text("{}")
    with pytest.raises(FileExistsError, match="allow_overwrite"):
        fit_cfm.main.__wrapped__(config)
