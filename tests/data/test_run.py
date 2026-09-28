import json

import pandas as pd
import torch
from anndata import AnnData

from sckitflow import load_run, run_rngs, save_run
from sckitflow.core.methods.inference._ode import ODEConfig
from sckitflow.core.methods.training._cfm import CFMConfig
from sckitflow.data._config import FlowDataConfig
from sckitflow.data._group_encoders import OneHot
from sckitflow.data.splitters._combination import CombinationSplitterConfig

SPLITTER = CombinationSplitterConfig(
    params={
        "group_keys": ("cell_line", "drug"),
        "always_train_keys": ("cell_line",),
        "control_key": "drug",
        "test_fraction": 0.5,
    }
)
DATA = FlowDataConfig(
    conditions={"drug": ("drug",)},
    conditions_reps={"drug": "drug"},
    groups=("cell_line",),
    groups_encoding={"cell_line": OneHot()},
    batch_size=4,
)


def _split(seed: int, split_seed: int, adata: AnnData) -> pd.Series:
    return SPLITTER.build(rng=run_rngs(seed=seed, split_seed=split_seed).split).assign(adata)


def test_split_depends_only_on_split_seed(adata_small: AnnData):
    pd.testing.assert_series_equal(_split(0, 3, adata_small), _split(1, 3, adata_small))


def test_load_run_round_trips(tmp_path, adata_small: AnnData):
    module = torch.nn.Linear(2, 2)
    save_run(
        tmp_path,
        seed=0,
        split_seed=3,
        splitter_config=SPLITTER,
        data_config=DATA,
        method_configs={"training": CFMConfig(), "inference": ODEConfig(params={"n_steps": 5})},
        module=module,
    )
    document = json.loads((tmp_path / "specs.json").read_text())
    assert document["seeds"] == {"seed": 0, "split_seed": 3}
    assert document["data"]["config"]["groups_encoding"]["cell_line"]["type"] == "group_encoder.one_hot"

    datamodule, plan = load_run(tmp_path, adata_small, torch.nn.Linear(2, 2))
    assert plan.training_method.module.weight.equal(module.weight)
    assert "test" in datamodule.val_names
