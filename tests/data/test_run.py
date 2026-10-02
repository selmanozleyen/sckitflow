import json

import torch
from anndata import AnnData

from sckitflow import RunConfig, load_run, save_run
from sckitflow.core.methods.inference._ode import ODEInferenceConfig
from sckitflow.core.methods.training._cfm import CFMTrainingConfig
from sckitflow.data._config import FlowDataModuleConfig
from sckitflow.data._group_encoders import OneHotEncoderConfig
from sckitflow.data.splitters._combination import CombinationSplitterConfig

SPLITTER = CombinationSplitterConfig(
    group_keys=("cell_line", "drug"), always_train_keys=("cell_line",), control_key="drug", test_fraction=0.5
)
DATA = FlowDataModuleConfig(
    conditions={"drug": ("drug",)},
    conditions_reps={"drug": "drug"},
    groups=("cell_line",),
    groups_encoding={"cell_line": OneHotEncoderConfig()},
    batch_size=4,
)


def test_load_run_round_trips(tmp_path, adata_small: AnnData):
    module = torch.nn.Linear(2, 2)
    spec = RunConfig(
        data=DATA,
        training=CFMTrainingConfig(),
        inference=ODEInferenceConfig(n_steps=5),
        splitter=SPLITTER,
        splitter_seed=3,
    )
    save_run(tmp_path, spec, module)
    document = json.loads((tmp_path / "specs.json").read_text())
    assert (document["loader_seed"], document["splitter_seed"]) == (0, 3)
    assert document["data"]["groups_encoding"]["cell_line"]["type"] == "group_encoder.one_hot"

    datamodule, plan = load_run(tmp_path, adata_small, torch.nn.Linear(2, 2))
    assert plan.training_method.module.weight.equal(module.weight)
    assert "test" in datamodule.val_names
