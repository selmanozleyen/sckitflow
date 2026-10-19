# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Changed

- `save_run` / `load_run` are replaced by `Run.save_pretrained` / `Run.from_pretrained`, in the Hugging Face Hub layout of `scfit.pretrained` (`config.json` plus `module/model.safetensors`); `from_pretrained` also takes a Hub repo id. A `Run` now carries its `spec`. Specs carry no `version`. Runs and checkpoints saved before this no longer load: a run reads back under the package versions that saved it

### Added

- Basic tool, preprocessing and plotting functions
