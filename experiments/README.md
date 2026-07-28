# Code and dataset for the experiment

This subdirectory contains the experiment code and dataset from the MetricSifter paper.

## Download dataset

Run the following commands from the repository root. The loaders read the downloaded
`*.tar.bz2` archives directly, so do not extract them. They must remain in
`experiments/data/`.

```shell-session
mkdir -p experiments/data
gh release download data-v1.0.0 -D experiments/data --pattern "*.tar.bz2"
```

The expected files are `synthetic_data.tar.bz2`, `ss-small.tar.bz2`,
`ss-medium.tar.bz2`, `ss-large.tar.bz2`, `tt-small.tar.bz2`,
`tt-medium.tar.bz2`, and `tt-large.tar.bz2`.

## Setup Python Environment

Install Python via [pyenv](https://github.com/pyenv/pyenv) or any other way you like.

```shell-session
pyenv install 3.10.14
pyenv local 3.10.14
```

Run the following commands from the repository root (the parent directory of this
`experiments` directory). PyRCA pins scikit-learn below 1.2, while the standard
development environment uses a current scikit-learn release. Keep experiments in a
separate Python 3.10 virtual environment; the `dev` and `experiments` extras are
intentionally declared as conflicting.

```shell-session
uv venv --python 3.10 .venv-experiments
uv pip install --python .venv-experiments/bin/python -e ".[experiments]"
```

Only the localization experiments require PyRCA. It is not available on PyPI, so
install its pinned revision into the experiment environment when running those experiments:

```shell-session
uv pip install --python .venv-experiments/bin/python \
  "sfr-pyrca @ git+https://github.com/salesforce/PyRCA@d85512b"
```

For a pip-based setup, `experiments/requirements.txt` is intentionally a thin wrapper
around the same project extra plus the pinned PyRCA revision. Create another Python 3.10
environment and install it from the repository root so that `-e ".[experiments]"` refers
to this checkout:

```shell-session
python3.10 -m venv .venv-experiments
.venv-experiments/bin/python -m pip install -r experiments/requirements.txt
```

## Run experiment

The experiment modules use imports relative to the `experiments` directory. After the
environment and data are ready, run this example from that directory to reproduce the
synthetic reduction sweep and write its raw scores:

```shell-session
cd experiments
../.venv-experiments/bin/python - <<'PY'
from dataset.loader import load_synthetic_data
from sweeper.sweeper import sweep_reduction_on_synthetic

dataset = load_synthetic_data()
scores = sweep_reduction_on_synthetic(dataset, n_jobs=-1)
scores.to_csv("synthetic-reduction-scores.csv", index=False)
PY
```

The same loader and sweeper modules provide the empirical entry points:

```python
from dataset.loader import load_sockshop_data, load_trainticket_data
from sweeper.sweeper import sweep_reduction_on_empirical

sockshop_scores = sweep_reduction_on_empirical(load_sockshop_data(), n_jobs=-1)
trainticket_scores = sweep_reduction_on_empirical(load_trainticket_data(), n_jobs=-1)
```

For interactive exploration, the existing
`experiments/notebooks/synthetic_reduction.ipynb` notebook calls the same
`load_synthetic_data()` and `sweep_reduction_on_synthetic()` functions. Jupyter is not
a project dependency; use a separately managed Jupyter installation with this
environment as its kernel. There is no separate experiment CLI.
