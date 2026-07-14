# Code and dataset for the experiment

This subdirectory contains the experiment code and dataset from the MetricSifter paper.

## Download dataset

You can download the dataset via the following command from GitHub releases.

```shell-session
gh release download data-v1.0.0 -D data/ --pattern "*.tar.bz2"
```

## Setup Python Environment

Install Python via [pyenv](https://github.com/pyenv/pyenv) or any other way you like.

```shell-session
pyenv install 3.11.6
pyenv local 3.11.6
```

Run the following commands from the repository root (the parent directory of this `experiments` directory). The
editable install uses the current MetricSifter checkout rather than a remote `main` branch. PyRCA is installed
separately from its pinned Git revision because it is not available on PyPI.

```shell-session
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiments]"
python -m pip install "sfr-pyrca @ git+https://github.com/salesforce/PyRCA@d85512b"
```

## Run experiment
