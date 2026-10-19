# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Changed

- `FlowDataModule` holds its `adata` for its whole life: `attach`, `set_predict_data`, `predict_dataloader` and its `control_adata` are removed. `load_from_checkpoint(path, adata)` builds it with the data, and `predict_adata` builds its eval loader per call.

### Added

- Basic tool, preprocessing and plotting functions
- `docs/scripts/fit_cfm.py` trains a run from one YAML spec; install with the `scripts` extra
- `RunConfig.build` and `load_run` take `control_adata`, and `RunConfig.build` takes `metrics`

### Changed

- Flow method configs take only configs: live probability paths and raw sampler callables are rejected
- Time and noise samplers are separate families (`TimeSamplerConfig`, `NoiseSamplerConfig`)
