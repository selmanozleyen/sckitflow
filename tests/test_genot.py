"""GENOT across state spaces: the sources' state is their own rep (`source_sample_rep`), matched by Gromov-Wasserstein."""

import logging

import lightning as pl
import numpy as np
import pytest
import torch

from sckitflow import Model, ModelConfig, predict_adata
from sckitflow.core.coupling import GromovWassersteinCouplingConfig
from sckitflow.core.methods import CFMTrainingConfig, MatchedTrainingConfig, ODEInference, ODEInferenceConfig
from sckitflow.core.nn import MLPVelocityConfig
from sckitflow.data import DataManagerConfig, FlowDataModule
from sckitflow.data._group_encoders import LabelEncoderConfig
from sckitflow.data.sim import get_dummy_adata
from sckitflow.trainer import TrainingPlan

ENCODER = {"source_encoder_mlp_kwargs": {"hidden_dims": [8]}, "source_encoder_output_dim": 8}


def _two_spaces(source_width: int = 4):
    """Controls whose state is `X_atac`, perturbed cells whose state is `X_rna` (6 wide); each file only has its own."""
    adata = get_dummy_adata(
        n_obs_pert=120,
        n_obs_ctrl=40,
        n_genes=10,
        obsm_keys_to_dim={},
        obs_columns_to_nunique_and_prefix={"drug": (3, "drug"), "cell_line": (2, "cell_line")},
        uns_keys_to_nunique_prefix_and_dim={"drug": (3, "drug", 4)},
        obs_columns_to_fixed_val={"drug": "control"},
        rng=np.random.default_rng(0),
    )
    adata.obs_names = adata.obs_names.astype(str)
    rng = np.random.default_rng(1)
    is_ctrl = (adata.obs["drug"] == "control").to_numpy()
    ctrl, pert = adata[is_ctrl].copy(), adata[~is_ctrl].copy()
    ctrl.obsm["X_atac"] = rng.normal(size=(ctrl.n_obs, source_width)).astype("float32")
    pert.obsm["X_rna"] = rng.normal(size=(pert.n_obs, 6)).astype("float32")
    return ctrl, pert


def _spec(*, generate_from_noise: bool = True, module: dict | None = None) -> ModelConfig:
    return ModelConfig(
        data=DataManagerConfig(
            sample_rep="X_rna",
            source_sample_rep="X_atac",
            categorical_conditions={"drug": ("drug",)},
            categorical_reps={"drug": "drug"},
            control_values_dict={"drug": "control"},
            groups=("cell_line",),
            groups_encoding={"cell_line": LabelEncoderConfig(classes=("cell_line0", "cell_line1"))},
            source_rep_quad="X_atac",
            target_rep_quad="X_rna",
            use_linear=False,
        ),
        module=MLPVelocityConfig(**(ENCODER if module is None else module)),
        training=MatchedTrainingConfig(
            method=CFMTrainingConfig(generate_from_noise=generate_from_noise),
            # scaled costs: unscaled float32 squared distances make entropic GW underflow to a degenerate plan
            coupling=GromovWassersteinCouplingConfig(scale_cost="mean", epsilon=0.1),
        ),
        inference=ODEInferenceConfig(generate_from_noise=generate_from_noise, n_samples=1, n_steps=3),
    )


def test_genot_maps_between_spaces(tmp_path, caplog):
    ctrl, pert = _two_spaces()
    model = _spec().build([ctrl, pert])
    assert (model.dims.state_dim, model.dims.source_state_dim) == (6, 4)
    datamodule = FlowDataModule(model.manager, model.dims, [ctrl, pert], batch_size=8, n_train_steps=3)
    plan = TrainingPlan(model.training_method, torch.optim.Adam(model.module.parameters(), lr=1e-3))
    with caplog.at_level(logging.WARNING):
        pl.Trainer(
            max_steps=3, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False
        ).fit(plan, datamodule=datamodule)
    assert "uniform coupling" not in caplog.text  # the GW plan was real, not the uniform fallback

    model.save_pretrained(tmp_path)
    loaded = Model.from_pretrained(tmp_path, [ctrl, pert])
    for key, weight in model.module.state_dict().items():
        assert loaded.module.state_dict()[key].equal(weight)
    pred = predict_adata(datamodule, loaded.inference_method, [ctrl, pert], max_per_group=4)
    assert pred.n_vars == 6 and np.isfinite(np.asarray(pred.X)).all()  # predictions live in the targets' space


def test_two_spaces_of_the_same_width_are_told_apart():
    ctrl, pert = _two_spaces(source_width=6)  # ATAC and RNA both 6 wide: only `source_sample_rep` tells them apart
    assert _spec().build([ctrl, pert]).dims.source_state_dim == 6
    with pytest.raises(ValueError, match=r"own state space.*set `generate_from_noise=True`"):
        _spec(generate_from_noise=False).build([ctrl, pert])


def test_inference_alone_refuses_to_start_from_another_space():
    ctrl, pert = _two_spaces()
    model = _spec().build([ctrl, pert])
    with pytest.raises(ValueError, match="own state space"):
        ODEInference(model.module, ODEInferenceConfig(generate_from_noise=False), dims=model.dims)


@pytest.mark.parametrize(
    ("module", "match"),
    [
        ({}, "only reads through a source encoder"),
        ({"source_encoder_mlp_kwargs": {"input_dim": 4}}, "comes from the data"),
    ],
)
def test_the_source_encoder_reads_the_other_space(module, match):
    ctrl, pert = _two_spaces()
    with pytest.raises(ValueError, match=match):
        _spec(module=module).build([ctrl, pert])


def test_reloading_needs_target_cells(tmp_path):
    ctrl, pert = _two_spaces()
    _spec().build([ctrl, pert]).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="None of the AnnData holds target cells"):
        Model.from_pretrained(tmp_path, ctrl)
