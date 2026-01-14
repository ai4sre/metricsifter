# MetricSifter: Feature Reduction of Multivariate Time Series Data for Efficient Fault Localization in Cloud Applications
![CI workflow](https://github.com/ai4sre/metricsifter/actions/workflows/ci.yaml/badge.svg)

This repository contains code and datasets used in the experiments described in our paper [1].

- [1]: Yuuki Tsubouchi, Hirofumi Tsuruta, ["MetricSifter: Feature Reduction of Multivariate Time Series Data for Efficient Fault Localization in Cloud Applications"](https://doi.org/10.1109/ACCESS.2024.3374334), IEEE Access (ACCESS) , Vol. 12, pp. 37398-37417, March 2024.

## Introduction

MetricSifter is a feature reduction framework designed to accurately identify anomalous metrics caused by faults for enhancing fault localization. Our key insight is that the change point times inside the failure duration are close to each other for the failure-related metrics. MetricSifter detects change points per metric, localizes the time frame with the highest change point density, and excludes metrics with no change points in that time frame. The offline change point detection is implemented by [ruptures](https://github.com/deepcharles/ruptures), and the segmentation of the detected change points is based on kernel density estimation (KDE).

## Installation

### Prerequisites

If you want to use `uv` (recommended for faster installation), install it first:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### From PyPI

You can install `metricsifter` package from PyPI:

```bash
# Using pip
pip install metricsifter

# Using uv (recommended for faster installation)
uv pip install metricsifter
```

### For Development

```bash
# Clone the repository
git clone https://github.com/ai4sre/metricsifter.git
cd metricsifter

# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev]"
```

## Getting Started

```python
from metricsifter.sifter import Sifter
from tests.sample_gen.generator import generate_synthetic_data

## Create time series data
normal_data, abonormal_data, _, _, anomalous_nodes = generate_synthetic_data(num_node=20, num_edge=20, num_normal_samples=55, num_abnormal_samples=15, anomaly_type=0)
data = pd.concat([normal_data, abonormal_data], axis=0, ignore_index=True)

## Remove the variables of time series data
sifter = Sifter(penalty_adjust=2.0, n_jobs=1)
sifted_data = sifter.run(data=data)
print("(#removed metrics) / (#total metrics):", len(set(data.columns) - set(siftered_data.columns)), "/", len(data.columns))
print("difference between prediction and ground truth:", set(siftered_data.columns) - anomalous_nodes)
assert set(sifted_data.columns) - anomalous_nodes == set()
```

The example of original synthetic data and its sifted data is shown in the following figure.

### Before
<img src="./docs/images/original_time_series.png" width="600" height="480">

### After
<img src="./docs/images/sifted_time_series.png" width="600" height="360">


## For Developers

### Setup Development Environment

```bash
# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest -s -v tests
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .
```

## License

[BSD-3-Clause](LICENSE)
