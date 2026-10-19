from typing import get_args

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import ValidationError

from sckitflow._types import ActivationId
from sckitflow.core.nn import MLPVelocityConfig
from sckitflow.core.nn._config import _ACTIVATIONS


def test_mlp_settings_are_typed():
    with pytest.raises(ValidationError, match="hiden_dims"):
        MLPVelocityConfig(state_encoder_mlp_kwargs={"hiden_dims": [8]})


def test_hydra_config_round_trips_through_the_spec():
    hydra = OmegaConf.create({"state_encoder_mlp_kwargs": {"hidden_dims": [8, 8], "activation_cls": "silu"}})
    config = MLPVelocityConfig(state_encoder_mlp_kwargs=hydra.state_encoder_mlp_kwargs)
    assert MLPVelocityConfig.from_spec(config.to_spec()) == config


def test_activations_are_named_and_checked_at_load():
    config = MLPVelocityConfig(vf_decoder_mlp_kwargs={"hidden_dims": [8], "activation_cls": "silu"})
    assert config.vf_decoder_mlp_kwargs.mlp_kwargs()["activation_cls"] is torch.nn.SiLU
    with pytest.raises(ValidationError, match="activation_cls"):
        MLPVelocityConfig(vf_decoder_mlp_kwargs={"activation_cls": "Identity"})


def test_every_activation_id_has_a_class():
    assert set(get_args(ActivationId)) == set(_ACTIVATIONS)
