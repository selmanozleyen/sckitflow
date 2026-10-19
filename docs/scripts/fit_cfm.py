"""Full training pipeline for a Conditional Flow Matching (CFM) model.

Wires together the components needed to train a Conditional Flow Matching
model on single-cell data with an MLP-parameterised velocity field. Designed
to be invoked via Hydra as the entry point of a training run:

    python -m <package>.fit_cfm [+config overrides]

(e.g. ``python -m sckitflow.training.fit_cfm`` — the ``-m`` flag takes a
module path, not a file; adjust to the module's actual location.)

Pipeline (in the order ``main`` executes them)
----------------------------------------------
1. Seed everything from ``config.run.method_seed`` (``lightning.seed_everything``).
2. Load AnnData from disk (``read_adatas``; paths in ``config.paths``).
3. Build a ``CombinationSplitter`` from ``config.splitter`` (control/train/val/test
   splits) and a ``FlowDataModule`` from ``config.datamanager`` + ``config.datamodule``.
4. Build the MLP velocity field (``MLPVelocityConfig`` from ``config.module``).
5. Build the base training method (``CFMTrainingConfig``), wrap it in a
   ``MatchedTrainingConfig`` that pairs source/target samples according to
   ``config.coupling``, and build the inference method (``ODEInferenceConfig``).
   Probability path and samplers arrive as portable ``*Config`` objects so
   ``save_run`` can serialize them; each method resolves them to a callable
   inside ``build``.
6. Build an optimizer (``OptimizerConfig``) and validation metrics
   (``EnergyDistance``, ``MaximumMeanDiscrepancy``) from ``config.metrics``.
7. Build a ``TrainingPlan`` (the ``LightningModule`` wrapper) from
   ``config.training_plan``, wiring together the training/inference methods,
   metrics, optimizer, and transforms.
8. Configure ``WandbLogger`` + ``CSVLogger``, callbacks (``EarlyStopping``,
   ``GradientAccumulationScheduler``, ``LearningRateMonitor``, ``ModelCheckpoint``),
   and the ``lightning.pytorch.Trainer``.
9. ``trainer.fit(...)``.
10. Persist the run via ``save_run`` (writes ``specs.json`` + ``weights.pt``).
    Training checkpoints are written separately by the ``ModelCheckpoint``
    callback during ``trainer.fit``.
11. Return the last validation ``edist`` from ``trainer.callback_metrics``
    (fallback: ``+inf``) for sweeps / shell exit codes. The key is
    ``f"{val_name}/edist"`` where ``val_name`` is the first entry of
    ``datamodule.val_names`` (default ``"val"`` → ``"val/edist"``).

Key concepts
------------
- **Conditional Flow Matching (CFM)**: learns a velocity field v(x, t | c) that
  transports a source distribution to a target distribution conditioned on c.
- **Probability path**: defines the interpolation between source and target
  samples. Supported: ``linear_dirac``, ``linear_gaussian``,
  ``schrodinger_bridge``, ``variance_preserving``. See
  ``sckitflow.core.probability_paths``.
- **Velocity field**: the network that predicts the time derivative along the
  path. Here an MLP with separate state/time/condition encoders.
- **Coupling**: at training time, pairs source samples with target samples
  before regression. Selected by ``config.coupling.method``: ``independent``,
  ``exact``, ``sinkhorn``, ``partial``, ``unbalanced``,
  ``gromov-wasserstein``, ``fused``. See ``sckitflow.core.coupling``.
- **Sampler**: time and noise draws during training / inference.
  ``time_sampler`` and ``noise_sampler`` in the config are ``SamplerConfig``
  objects (portable — the run's spec serializes them). See
  ``sckitflow.core._samplers``.
- **Inference**: integrates the learned ODE from t=0 to t=1 with ``torchdiffeq``,
  configured via ``ODEInferenceConfig``.

Configuration
-------------
The full YAML is expected at ``<module_dir>/../config/fit_cfm.yaml`` (Hydra).
See that file for every key, its meaning, and its default. The coupling
selection and its hyperparameters live under ``config.coupling``.
``hydra.job.chdir: false`` is recommended so relative paths
(``save.run_dir``, ``callbacks.model_checkpoint.dirpath``, logger
``save_dir``) resolve from the launch directory rather than Hydra's
``outputs/<date>/<time>/`` tree.

Outputs
-------
- Checkpoints: ``config.callbacks.model_checkpoint.dirpath`` (written by the
  ``ModelCheckpoint`` callback during training; best-``val_edist`` weights
  plus ``last.ckpt``).
- Loggers: ``config.loggers.{wandb,csv}.save_dir``.
- Serialised run: ``config.save.run_dir`` (via ``save_run``; contains
  ``specs.json`` — the portable ``RunConfig`` — and ``weights.pt``). Suitable
  for reloading with ``sckitflow.load_run`` for **inference**; a reloaded
  plan carries no validation metrics, so it is not suitable for re-training
  with validation.
- Return value: last ``<val_name>/edist`` from
  ``trainer.callback_metrics`` (``+inf`` if validation never ran).
"""

from typing import Any

import anndata as ad
import hydra
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import numpy as np
from omegaconf import DictConfig, OmegaConf

from sckitflow import RunConfig, get_logger, save_run
from sckitflow.core._samplers import (
    IsotropicGaussianNoiseSamplerConfig,
    SamplerConfig,
    UniformTimeSamplerConfig,
)
from sckitflow.core.coupling import (
    CouplingConfig,
    IndependentCouplingConfig,
    ExactCouplingConfig,
    SinkhornCouplingConfig,
    PartialCouplingConfig,
    UnbalancedCouplingConfig,
    GromovWassersteinCouplingConfig,
    FusedGromovWassersteinCouplingConfig,
)
from sckitflow.core.methods import CFMTrainingConfig, MatchedTrainingConfig, ODEInferenceConfig
from sckitflow.core.metrics import EnergyDistance, MaximumMeanDiscrepancy
from sckitflow.core.nn import MLPVelocityConfig
from sckitflow.core.probability_paths import (
    ProbabilityPathConfig,
    LinearDiracProbabilityPathConfig,
    LinearGaussianProbabilityPathConfig,
    SchrodingerBridgeProbabilityPathConfig,
    VariancePreservingDiracProbabilityPathConfig,
)
from sckitflow.data import CombinationSplitterConfig, DataManagerConfig, FlowDataModuleConfig
from sckitflow.trainer import TrainingPlan, OptimizerConfig


logger = get_logger(__name__)

EDIST_KEY = "edist"
MMD_KEY = "mmd"


def read_adatas(config: DictConfig) -> tuple[ad.AnnData, ad.AnnData | None]:
    """Reads the input data from disk.

    Returns a tuple of ``(target_adata, control_adata)``; ``target_adata`` is always
    read from the ``config.paths.target_adata`` field, while the ``control_adata`` is
    only returned when ``config.paths.control_adata`` is not ``None``. Otherwise,
    ``control_adata`` will be ``None`` and will directly be inferred as a subset of
    ``target_adata``, if the datamodule is configured in a case-control setting.
    """
    logger.info(f"Reading target adata from {config.paths.target_adata}.")
    target_adata = ad.read_h5ad(config.paths.target_adata)
    logger.info(f"Target adata read: {target_adata}")
    if config.paths.control_adata is not None:
        logger.info(f"Loading control adata from {config.paths.control_adata}.")
        control_adata = ad.read_h5ad(config.paths.control_adata)
        logger.info(f"Control adata read: {control_adata}")
    else:
        logger.info("No control adata provided, skipping.")
        control_adata = None
    return target_adata, control_adata


def get_probability_path_cfg(config: DictConfig) -> ProbabilityPathConfig:
    """Retrieves the probability path configuration object from the given specification."""
    if config.probability_path.name == "linear_dirac":
        return LinearDiracProbabilityPathConfig(sigma=config.probability_path.sigma)
    elif config.probability_path.name == "linear_gaussian":
        return LinearGaussianProbabilityPathConfig(sigma=config.probability_path.sigma)
    elif config.probability_path.name == "schrodinger_bridge":
        return SchrodingerBridgeProbabilityPathConfig(
            sigma=config.probability_path.sigma, eps=config.probability_path.eps
        )
    elif config.probability_path.name == "variance_preserving":
        return VariancePreservingDiracProbabilityPathConfig(sigma=config.probability_path.sigma)
    else:
        raise ValueError(f"Invalid probability path name {config.probability_path.name}.")


def get_time_sampler_cfg(config: DictConfig) -> SamplerConfig:
    """Retrieves the time-sampler *config* for CFM training.

    The returned ``SamplerConfig`` is portable — ``save_run`` can serialize it
    into ``specs.json`` — while the method that consumes it resolves it to a
    raw callable in ``AbstractFlowMethod.__init__``. At the moment, the only
    available time sampler is ``"uniform"``.
    """
    if config.time_sampler.name == "uniform":
        return UniformTimeSamplerConfig()
    else:
        raise ValueError(f"Invalid time sampler name {config.time_sampler.name}.")


def get_noise_sampler_cfg(config: DictConfig) -> SamplerConfig:
    """Retrieves the noise-sampler *config* for CFM training.

    Same rationale as :func:`get_time_sampler_cfg`. At the moment, the only
    available noise sampler is ``"isotropic_gaussian"``.
    """
    if config.noise_sampler.name == "isotropic_gaussian":
        return IsotropicGaussianNoiseSamplerConfig()
    else:
        raise ValueError(f"Invalid noise sampler name {config.noise_sampler.name}.")


def get_coupling_cfg(config: DictConfig) -> CouplingConfig:
    """Resolves the coupling configuration from the input."""
    if config.coupling.method == "independent":
        return IndependentCouplingConfig()
    elif config.coupling.method == "exact":
        return ExactCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
        )
    elif config.coupling.method == "sinkhorn":
        return SinkhornCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
            reg=config.coupling.reg,
            stop_thr=config.coupling.stop_thr,
        )
    elif config.coupling.method == "partial":
        return PartialCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
            reg=config.coupling.reg,
            stop_thr=config.coupling.stop_thr,
            m=config.coupling.m,
        )
    elif config.coupling.method == "unbalanced":
        return UnbalancedCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
            reg=config.coupling.reg,
            stop_thr=config.coupling.stop_thr,
            reg_m=config.coupling.reg_m,
        )
    elif config.coupling.method == "gromov-wasserstein":
        return GromovWassersteinCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            epsilon=config.coupling.epsilon,
            max_iter=config.coupling.max_iter,
            tol=config.coupling.tol,
        )
    elif config.coupling.method == "fused":
        return FusedGromovWassersteinCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            epsilon=config.coupling.epsilon,
            max_iter=config.coupling.max_iter,
            tol=config.coupling.tol,
            alpha=config.coupling.alpha,
        )
    else:
        raise ValueError(f"Invalid coupling method {config.coupling.method}")


def resolve_omegaconf_to_dictionary(conf_dict: dict[str, Any] | None | DictConfig) -> dict[str, Any]:
    """Resolves the omegaconf ``DictConfig`` to a plain dictionary."""
    out_dict = {}
    if conf_dict is not None:
        out_dict = conf_dict
        if isinstance(out_dict, dict):
            out_dict = OmegaConf.create(out_dict)
        out_dict = OmegaConf.to_container(out_dict, resolve=True)
    return out_dict


@hydra.main(config_path="../config", config_name="fit_cfm", version_base=None)
def main(config: DictConfig) -> float:
    """Main entrypoint for CFM training."""
    # Seed everything with lightning
    # This is the master seeding, and it will be also saved to the run
    # for further reproducibility.
    logger.info(f"Setting the global run seed to {config.run.method_seed}")
    seed_everything(seed=config.run.method_seed, workers=True)

    # Load data from disk using the utility function.
    # Please override this function to allow custom operations on the
    # data after loading; at the moment, only vanilla loading
    # with ``anndata`` is performed, and no other transformation is
    # applied to the data.
    target_adata, control_adata = read_adatas(config)

    # Prepare splitter for training.
    # The splitter defines control, train, validation and test splits
    # and writes a column in the input ``ad.AnnData``. We should always
    # fit any data transformation only on the training cases. At the moment,
    # no transformation is applied, so everything works -- but we need to
    # leave it here as a TODO for the future: once the splitter is fitted
    # on the data, fit the transformations on the training data only.
    logger.info(f"Initializing splitter with seed {config.run.splitter_seed}.")
    splitter_rng = np.random.default_rng(config.run.splitter_seed)
    splitter_cfg = CombinationSplitterConfig(
        group_keys=config.splitter.group_keys,
        always_train_keys=config.splitter.always_train_keys,
        control_key=config.splitter.control_key,
        control_value=config.splitter.control_value,
        test_fraction=config.splitter.test_fraction,
        train_label=config.splitter.train_label,
        test_label=config.splitter.test_label,
        control_label=config.splitter.control_label,
    )
    splitter = splitter_cfg.build(rng=splitter_rng)

    # Prepare data manager.
    # The ``DataManager`` is never materialized directly here: it lives
    # in the ``datamodule``, of which it is an attribute.
    logger.info("Initializing datamanager.")
    dm_cfg = DataManagerConfig(
        sample_rep=config.datamanager.sample_rep,
        conditions=config.datamanager.conditions,
        conditions_reps=config.datamanager.conditions_reps,
        conditions_covariates=config.datamanager.conditions_covariates,
        control_values_dict=config.datamanager.control_values_dict,
        matched_pairs=config.datamanager.matched_pairs,
        # split_by=config.datamanager.split_by,
        condition_state_key=config.datamanager.condition_state_key,
        target_categorical_covs_dict=config.datamanager.target_categorical_covs_dict,
        target_continuous_covs=config.datamanager.target_continuous_covs,
        groups=config.datamanager.groups,
        groups_reps=config.datamanager.groups_reps,
        groups_encoding=config.datamanager.groups_encoding,
        rep_lin=config.datamanager.rep_lin,
        source_rep_quad=config.datamanager.source_rep_quad,
        target_rep_quad=config.datamanager.target_rep_quad,
        use_linear=config.datamanager.use_linear,
    )

    # Initialize datamodule.
    # We set the datamodule-specific generator for reproducibility.
    # The datamodule configuration takes the datamanager_cfg and
    # provides a builder for the ``pl.DataModule`` wrapper.
    logger.info(f"Initializing datamodule with seed {config.run.loader_seed}")
    datamodule_rng = np.random.default_rng(config.run.loader_seed)
    datamodule_cfg = FlowDataModuleConfig(
        manager=dm_cfg,
        train_split=config.datamodule.train_split,
        n_train_steps=config.datamodule.n_train_steps,
        batch_size=config.datamodule.batch_size,
        dtype=config.datamodule.dtype,
        loader_kwargs=resolve_omegaconf_to_dictionary(config.datamodule.loader_kwargs),
    )
    datamodule = datamodule_cfg.build(
        target_adata,
        rng=datamodule_rng,
        control_adata=control_adata,
        splitter=splitter,
    )

    # Initialize neural network.
    # The dimensionalities are automatically inferred from the
    # data_dims object, linked to the datamodule.
    logger.info("Initializing neural velocity field.")
    module_cfg = MLPVelocityConfig(
        encode_state=config.module.encode_state,
        encode_time=config.module.encode_time,
        time_features_id=config.module.time_features_id,
        num_time_features=config.module.num_time_features,
        max_period=config.module.max_period,
        state_encoder_output_dim=config.module.state_encoder_output_dim,
        time_encoder_output_dim=config.module.time_encoder_output_dim,
        state_encoder_mlp_kwargs=config.module.state_encoder_mlp_kwargs,
        time_encoder_mlp_kwargs=config.module.time_encoder_mlp_kwargs,
        vf_decoder_mlp_kwargs=config.module.vf_decoder_mlp_kwargs,
        conditioning_id=config.module.conditioning_id,
        conditioning_kwargs=resolve_omegaconf_to_dictionary(config.module.conditioning_kwargs),
        condition_encoder_input_layers=config.module.condition_encoder_input_layers,
        condition_encoder_output_dim=config.module.condition_encoder_output_dim,
        condition_encoder_pooling_mode=config.module.condition_encoder_pooling_mode,
        condition_encoder_pooling_kwargs=resolve_omegaconf_to_dictionary(
            config.module.condition_encoder_pooling_kwargs
        ),
        condition_encoder_pooling_proj_dim=config.module.condition_encoder_pooling_proj_dim,
        condition_encoder_pooling_proj_bias=config.module.condition_encoder_pooling_proj_bias,
        condition_encoder_covariates_not_pooled=config.module.condition_encoder_covariates_not_pooled,
        condition_encoder_output_layers_kwargs=config.module.condition_encoder_output_layers_kwargs,
        source_encoder_mlp_kwargs=config.module.source_encoder_mlp_kwargs,
        source_encoder_output_dim=config.module.source_encoder_output_dim,
    )
    module = module_cfg.build(datamodule.data_dims)

    # Initialize probability path, time sampler and noise sampler.
    # This initialization is somewhat of a hack at the moment, we should
    # probably access the registered components from scfit; however, it is
    # not clear to me how to access the ``scfit.registry._REGISTRY`` with
    # ``scfit`` public API. TODO: Consider changing the initialization
    # of the probability path to use ``scfit`` components.
    # Instantiating probability path configuration.
    logger.info("Initializing probability path.")
    probability_path_cfg = get_probability_path_cfg(config)
    # Instantiating time sampler.
    logger.info("Initializing time sampler.")
    time_sampler_cfg = get_time_sampler_cfg(config)
    # Instantiating noise sampler.
    logger.info("Initializing noise sampler.")
    noise_sampler_cfg = get_noise_sampler_cfg(config)

    # Initialize training module.
    # In this script, we train a CFM model. This could be potentially
    # generalized to allow for arbitrary training components.
    # The pipeline would be quite similar for other models, and
    # the same steps cand be executed in the same order as here.
    logger.info("Initializing base training method.")
    base_training_cfg = CFMTrainingConfig(
        probability_path=probability_path_cfg,
        time_sampler=time_sampler_cfg,
        noise_sampler=noise_sampler_cfg,
        generate_from_noise=config.training.generate_from_noise,
    )

    # Initialize coupling configuration and matched training configurations.
    # The ``MatchedTrainingConfig`` wraps a base training configuration and a
    # coupling configuration to perform matching before calling the training method.
    # We wrap around the ``base_training_cfg`` that we defined just above.
    logger.info("Initializing coupling module.")
    coupling_cfg = get_coupling_cfg(config)
    logger.info("Initializing matched wrapper.")
    training_cfg = MatchedTrainingConfig(
        method=base_training_cfg,
        coupling=coupling_cfg,
    )

    #  We use the wrapped matched method in ``training_cfg`` to build the training protocol,
    # this ensures that the matching is correctly computed before each training step.
    logger.info("Building training method.")
    training_method = training_cfg.build(
        module,
        dims=datamodule.data_dims,
    )

    # Initialize inference module. This inference module is
    # used for the validation step during training. Note that
    # the computational complexity of the solver can significantly
    # affect the wall time, especially with high frequency validation
    # passes. It is hence highly reccomended to set configure it
    # as a cheap sampler (reasonably low ``n_steps``, with euler integration).
    # More accurate predictors modules can be then initialized at inference time,
    # without causing eccessive computational overhead.
    logger.info("Initializing inference method.")
    inference_cfg = ODEInferenceConfig(
        probability_path=probability_path_cfg,
        time_sampler=time_sampler_cfg,
        noise_sampler=noise_sampler_cfg,
        generate_from_noise=config.inference.generate_from_noise,
        solver_kwargs=resolve_omegaconf_to_dictionary(config.inference.solver_kwargs),
        return_trajectory=config.inference.return_trajectory,
        n_steps=config.inference.n_steps,
        n_samples=config.inference.n_samples,
    )
    logger.info("Building inference method.")
    inference_method = inference_cfg.build(module, dims=datamodule.data_dims)

    # Prepare optimizer and scheduler.
    # ``sckitflow`` provides a public API to handle optimizers; this is not the case
    # for the learning rate schedulers.
    # TODO: maybe consider providing the same utility wrappers for the lr schedulers,
    # so that the pipeline can be fully specified with the necessary configuration files.
    logger.info("Initializing optimizer.")
    optimizer_cfg = OptimizerConfig(
        name=config.optimizer.name,
        kwargs=resolve_omegaconf_to_dictionary(config.optimizer.kwargs),
    )
    logger.info("Building optimizer.")
    optimizer = optimizer_cfg.build(module.parameters())
    lr_scheduler = None

    # Initialize metrics used to evaluate model's preditions during validation.
    # We use two distributional metrics: energy distance and maximum mean discrepancy.
    # These metrics are useful in the unpaired setting, when there is not a direct and
    # immediate correspondence between source and target samples -- and we want to
    # compare probability distributions instead.
    logger.info("Initializing metrics.")
    edist = EnergyDistance(dist_sync_on_step=config.metrics.edist.dist_sync_on_step)
    mmd = MaximumMeanDiscrepancy(
        gammas=config.metrics.mmd.gammas,
        dist_sync_on_step=config.metrics.mmd.dist_sync_on_step,
    )
    # The trainer expects a dictionary with named metrics; we use the global constants to
    # define the respective keys for each metric.
    metrics = {
        EDIST_KEY: edist,
        MMD_KEY: mmd,
    }

    # Initialize prediction transformations.
    # TODO: Double check ``sckitflow`` API to verify how this transformation should be defined
    pred_transform = None  # TODO

    # Initialize target transformations.
    # TODO: Double check ``sckitflow`` API to verify how this transformation should be defined
    target_transform = None  # TODO

    # Initialize training plan. This still lives in the ``sckitflow`` namespace as a wrapper
    # around ``lightning`` API. All the ``lightning.pytorch.Trainer`` needs is this configured
    # plan and the ``datamodule`` we constructed before. After this, we only need to configure
    # general ``lightning``, as everything from the ``sckitflow`` side will be already prepared.
    logger.info("Initializing training plan.")
    plan = TrainingPlan(
        training_method,
        optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_interval=config.training_plan.lr_scheduler_interval,
        inference_method=inference_method,
        metrics=metrics,
        val_names=datamodule.val_names,
        pred_transform=pred_transform,
        target_transform=target_transform,
        seed=config.training_plan.seed,
    )

    # Prepare wandb and csv loggers. The loggers are then passed to the
    # ``lightning.pytorch.Trainer`` at initialization. Please note that
    # wandb needs to be installed and configured for this to work. The
    # logging objects live in the ``lightning.pytorch`` and are ``sckitflow``
    # indpendent. We use wandb for remote logging, and csv for local logging.
    # For additional loggers, check https://pytorch-lighting.readthedocs.io/en/stable/api_references.html#loggers.
    logger.info("Initializing wandb logger.")
    wandb_logger = WandbLogger(
        name=config.loggers.wandb.name,
        save_dir=config.loggers.wandb.save_dir,
        offline=config.loggers.wandb.offline,
        id=config.loggers.wandb.id,
        anonymous=config.loggers.wandb.anonymous,
        version=config.loggers.wandb.version,
        project=config.loggers.wandb.project,
        log_model=config.loggers.wandb.log_model,
        experiment=config.loggers.wandb.experiment,
        prefix=config.loggers.wandb.prefix,
        **resolve_omegaconf_to_dictionary(config.loggers.wandb.init_kwargs),
    )
    logger.info("Initializing csv logger.")
    csv_logger = CSVLogger(
        config.loggers.csv.save_dir,
        name=config.loggers.csv.name,
        version=config.loggers.csv.version,
        prefix=config.loggers.csv.prefix,
        flush_logs_every_n_steps=config.loggers.csv.flush_logs_every_n_steps,
    )
    training_loggers = [wandb_logger, csv_logger]

    # Prepare callbacks. We use four callbacks at the moment, but this can
    # be easily extended in an analogous way. The used callbacks are: early stopping,
    # gradient accumulation, learning rate monitoring, and model checkpointing.
    # They all live in the ``lightning.pytorch.callbacks`` namespace. Visit
    # https://pytorch-lighting.readthedocs.io/en/stable/api_references.html#callbacks
    # for all the available callbacks and their respective construction parameters.
    # TODO: maybe consider registering the configurations for lightining callbacks as
    # components in ``scfit``, so that their configurations can be effectively serialized
    # for full end-to-end reproducibility.
    logger.info("Initializing early stopping callback.")
    early_stopping = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        min_delta=config.callbacks.early_stopping.min_delta,
        patience=config.callbacks.early_stopping.patience,
        verbose=config.callbacks.early_stopping.verbose,
        mode=config.callbacks.early_stopping.mode,
        strict=config.callbacks.early_stopping.strict,
        check_finite=config.callbacks.early_stopping.check_finite,
        stopping_threshold=config.callbacks.early_stopping.stopping_threshold,
        divergence_threshold=config.callbacks.early_stopping.divergence_threshold,
        check_on_train_epoch_end=config.callbacks.early_stopping.check_on_train_epoch_end,
    )
    logger.info("Initializing gradient accumulation callback.")
    grad_accumulation = GradientAccumulationScheduler(scheduling=config.callbacks.grad_accumulation.scheduling)
    logger.info("Initializing learning rate monitor callback.")
    lr_monitor = LearningRateMonitor(
        logging_interval=config.callbacks.lr_monitor.logging_interval,
        log_momentum=config.callbacks.lr_monitor.log_momentum,
    )
    logger.info("Initializing model checkpoint callback.")
    model_checkpoint = ModelCheckpoint(
        dirpath=config.callbacks.model_checkpoint.dirpath,
        filename=config.callbacks.model_checkpoint.filename,
        monitor=config.callbacks.model_checkpoint.monitor,
        verbose=config.callbacks.model_checkpoint.verbose,
        save_last=config.callbacks.model_checkpoint.save_last,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        save_weights_only=config.callbacks.model_checkpoint.save_weights_only,
        mode=config.callbacks.model_checkpoint.mode,
        auto_insert_metric_name=config.callbacks.model_checkpoint.auto_insert_metric_name,
        every_n_train_steps=config.callbacks.model_checkpoint.every_n_train_steps,
        train_time_interval=config.callbacks.model_checkpoint.train_time_interval,
        every_n_epochs=config.callbacks.model_checkpoint.every_n_epochs,
        save_on_train_epoch_end=config.callbacks.model_checkpoint.save_on_train_epoch_end,
    )
    callbacks = [early_stopping, grad_accumulation, lr_monitor, model_checkpoint]

    # Configuring the training plugings. This step is skipped for the moment and the
    # plugins are hard-coded to ``None``. They are here just as a reminder that they can be set,
    # and for us to consider whether to actually use them.
    # Visit https://pytorch-lighting.readthedocs.io/en/stable/api_references.html#plugins
    # for the available plugins. TODO: consider which plugins to utilize the plugins for training.
    plugins = None  # TODO

    # Initialize ``lightning.pytorch.Trainer`` object and set all the specified arguments.
    # The available arguments to initialize the Trainer can be verified from
    # https://pytorch-lighting.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer.
    # TODO: consider adding a ``TrainerConfig`` object to the ``sckitflow`` API for full end-to-end reproducibility.
    logger.info("Initializing trainer.")
    trainer = Trainer(
        accelerator=config.trainer.accelerator,
        strategy=config.trainer.strategy,
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes,
        precision=config.trainer.precision,
        logger=training_loggers,
        callbacks=callbacks,
        fast_dev_run=config.trainer.fast_dev_run,
        max_epochs=config.trainer.max_epochs,
        min_epochs=config.trainer.min_epochs,
        max_steps=config.trainer.max_steps,
        min_steps=config.trainer.min_steps,
        max_time=config.trainer.max_time,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
        limit_predict_batches=config.trainer.limit_predict_batches,
        overfit_batches=config.trainer.overfit_batches,
        val_check_interval=config.trainer.val_check_interval,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_checkpointing=config.trainer.enable_checkpointing,
        enable_progress_bar=config.trainer.enable_progress_bar,
        enable_model_summary=config.trainer.enable_model_summary,
        gradient_clip_val=config.trainer.gradient_clip_val,
        gradient_clip_algorithm=config.trainer.gradient_clip_algorithm,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        inference_mode=config.trainer.inference_mode,
        use_distributed_sampler=config.trainer.use_distributed_sampler,
        profiler=config.trainer.profiler,  # TODO: this can be an object, resolve it somehow.
        detect_anomaly=config.trainer.detect_anomaly,
        barebones=config.trainer.barebones,
        plugins=plugins,
        sync_batchnorm=config.trainer.sync_batchnorm,
        reload_dataloaders_every_n_epochs=config.trainer.reload_dataloaders_every_n_epochs,
        default_root_dir=config.trainer.default_root_dir,
        enable_autolog_hparams=config.trainer.enable_autolog_hparams,
        model_registry=config.trainer.model_registry,
    )
    # We call the ``Trainer.fit`` method by passing the ``datamodule``, instantiated
    # using the ``sckitflow`` API. We also pass the ``ckpt_path`` for model checkpointing.
    # TODO: consider how the ``ckpt_path`` interacts with the ``ModelCheckpoint`` and
    # possibly consider using only one of the two.
    logger.info("Fitting the model.")
    trainer.fit(plan, datamodule=datamodule, ckpt_path=config.trainer.ckpt_path)

    # Build run config and save to disk. This object contains all the configurations
    # currently defining a run. It is required for persistency and reproducibility, as
    # the ``save_run`` expects this object as input.
    logger.info("Initializing run configuration for reproducibility.")
    run_cfg = RunConfig(
        data=datamodule_cfg,
        training=training_cfg,
        inference=inference_cfg,
        splitter=splitter_cfg,
        loader_seed=config.run.loader_seed,
        splitter_seed=config.run.splitter_seed,
        method_seed=config.run.method_seed,
        module=module_cfg,
        optimizer=optimizer_cfg,
    )
    logger.info(f"Saving run {config.save.run_dir}.")
    save_run(config.save.run_dir, run_cfg, module, allow_overwrite=config.run.allow_overwrite)

    # Capture the edist from the latest validation step. This is the value
    # that will be returned to the caller (Hydra sweep / Optuna trial / shell
    # exit code). If validation never ran (fast_dev_run, limit_val_batches=0),
    # fall back to +inf so downstream comparators treat the run as the worst.
    # We guard ``dict.get`` with a ``float("inf")`` default to prevent ``KeyError``s
    # TODO: double check that the key which to retrieve the last metric from
    # is the correct one.
    val_edist_key = f"{datamodule.val_names[0]}/{EDIST_KEY}"  # e.g. "val/edist"
    last_val_edist = trainer.callback_metrics.get(val_edist_key, float("inf"))
    returned_metric = float(last_val_edist)
    logger.info(f"Finishing the run with edist of {returned_metric:.4f}")
    return returned_metric


if __name__ == "__main__":
    main()
