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

Install packages via the following commands.

```shell-session
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run experiment
