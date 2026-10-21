"""GENOT across spaces: sources and targets with different state widths, matched by Gromov-Wasserstein."""

import lightning as pl
import numpy as np
import pytest
import torch

from sckitflow import Model, ModelConfig, predict_adata
from sckitflow.core.coupling import GromovWassersteinCouplingConfig
from sckitflow.core.methods import CFMTrainingConfig, MatchedTrainingConfig, ODEInferenceConfig
from sckitflow.core.nn import MLPVelocityConfig
from sckitflow.data import DataManagerConfig, FlowDataModule
from sckitflow.data._group_encoders import LabelEncoderConfig
from sckitflow.data.sim import get_dummy_adata
from sckitflow.trainer import TrainingPlan


def _two_spaces():
    """Controls in a 4-wide space, perturbed cells in a 6-wide one, each with its own quadratic rep."""
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
    ctrl.obsm["X_state"] = ctrl.obsm["X_quad"] = rng.normal(size=(ctrl.n_obs, 4)).astype("float32")
    pert.obsm["X_state"] = pert.obsm["X_quad"] = rng.normal(size=(pert.n_obs, 6)).astype("float32")
    return ctrl, pert


def _spec(*, generate_from_noise: bool) -> ModelConfig:
    return ModelConfig(
        data=DataManagerConfig(
            sample_rep="X_state",
            categorical_conditions={"drug": ("drug",)},
            categorical_reps={"drug": "drug"},
            control_values_dict={"drug": "control"},
            groups=("cell_line",),
            groups_encoding={"cell_line": LabelEncoderConfig(classes=("cell_line0", "cell_line1"))},
            source_rep_quad="X_quad",
            target_rep_quad="X_quad",
            use_linear=False,
        ),
        module=MLPVelocityConfig(source_encoder_mlp_kwargs={"hidden_dims": [8]}, source_encoder_output_dim=8),
        training=MatchedTrainingConfig(
            method=CFMTrainingConfig(generate_from_noise=generate_from_noise),
            coupling=GromovWassersteinCouplingConfig(epsilon=1.0),
        ),
        inference=ODEInferenceConfig(generate_from_noise=generate_from_noise, n_samples=1, n_steps=3),
    )


def test_genot_maps_between_spaces(tmp_path):
    ctrl, pert = _two_spaces()
    model = _spec(generate_from_noise=True).build([ctrl, pert])
    assert (model.dims.state_dim, model.dims.source_state_dim) == (6, 4)
    datamodule = FlowDataModule(model.manager, model.dims, [ctrl, pert], batch_size=8, n_train_steps=3)
    plan = TrainingPlan(model.training_method, torch.optim.Adam(model.module.parameters(), lr=1e-3))
    pl.Trainer(max_steps=3, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False).fit(
        plan, datamodule=datamodule
    )

    model.save_pretrained(tmp_path)
    loaded = Model.from_pretrained(tmp_path, [ctrl, pert])
    pred = predict_adata(datamodule, loaded.inference_method, [ctrl, pert], max_per_group=4)
    assert pred.n_vars == 6 and np.isfinite(np.asarray(pred.X)).all()  # predictions live in the targets' space


def test_flowing_from_sources_needs_one_space():
    ctrl, pert = _two_spaces()
    with pytest.raises(ValueError, match=r"4 wide and the targets' 6.*set `generate_from_noise=True`"):
        _spec(generate_from_noise=False).build([ctrl, pert])
