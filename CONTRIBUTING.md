# Contributing

Thanks for helping with MetricSifter.

- **Discuss first** for anything beyond a small fix: open an issue that
  describes the problem or the idea, ideally with a small (synthetic) dataset
  that shows it.
- **Keep pull requests focused.** One change per PR, with tests for any
  behaviour change. The defaults of `Sifter` reproduce the published
  algorithm, so a new knob is opt-in, and a change to the default selection
  needs evidence beyond the unit tests.
- **Run `make check`** before pushing (`uv` and Go required). It verifies the
  lock file, runs ruff, black, mypy and the full test suite, builds the
  package and audits the workflows. Touching change-point detection, segmentation, the Prometheus
  adapter or CLI input handling? Run `make test-fuzz` too.
- **Keep the docs in sync.** A change to the public API, the CLI or the
  `SiftResult` report updates README.md.
- **Leave `experiments/` alone** unless the change is about the paper's
  experiments; it has its own environment (see `experiments/README.md`).
- **Commit style:** commits, branches and pull request titles follow the Git
  conventions in [AGENTS.md](AGENTS.md#git-conventions).

The conventions, the checks to run and the compatibility contracts are in
[AGENTS.md](AGENTS.md). Security issues go through [SECURITY.md](SECURITY.md),
not public issues.
