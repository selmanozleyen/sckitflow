import pytest
import torch
from omegaconf import OmegaConf
from pydantic import ValidationError

from sckitflow.core.nn import MLPVelocityConfig


def test_mlp_settings_are_typed():
    with pytest.raises(ValidationError, match="hiden_dims"):
        MLPVelocityConfig(state_encoder_mlp_kwargs={"hiden_dims": [8]})


def test_hydra_config_round_trips_through_the_spec():
    hydra = OmegaConf.create({"state_encoder_mlp_kwargs": {"hidden_dims": [8, 8], "activation_cls": "silu"}})
    config = MLPVelocityConfig(state_encoder_mlp_kwargs=hydra.state_encoder_mlp_kwargs)
    assert MLPVelocityConfig.from_spec(config.to_spec()) == config


def test_activation_class_is_stored_by_name_and_built_as_class():
    config = MLPVelocityConfig(vf_decoder_mlp_kwargs={"hidden_dims": [8], "activation_cls": torch.nn.SiLU})
    assert config.vf_decoder_mlp_kwargs.activation_cls == "SiLU"
    assert config.vf_decoder_mlp_kwargs.mlp_kwargs()["activation_cls"] is torch.nn.SiLU
