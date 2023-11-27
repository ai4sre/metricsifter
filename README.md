# MetricSifter
![CI workflow](https://github.com/ai4sre/metricsifter/actions/workflows/ci.yaml/badge.svg)

## Introduction

TBD

## Installation

You can install `metricsifter` package from PyPI via `pip install metricsifter`.

## Getting Started

```python
from metricsifter.sifter import Sifter
from tests.sample_gen.generator import generate_synthetic_data

## Prepare time series data
normal_data, abonormal_data, _, _, anomalous_nodes = generate_synthetic_data(num_node=20, num_edge=20, num_normal_samples=55, num_abnormal_samples=15, anomaly_type=0)
data = pd.concat([normal_data, abonormal_data], axis=0, ignore_index=True)

sifter = Sifter(penalty_adjust=2.0, n_jobs=1)
sifted_data = sifter.run(data=data)
print("(#removed metrics) / (#total metrics):", len(set(data.columns) - set(siftered_data.columns)), "/", len(data.columns))
print("difference between prediction and ground truth:", set(siftered_data.columns) - anomalous_nodes)
assert set(sifted_data.columns) - anomalous_nodes == set()
```

The example of original synthetic data and its sifted data is shown in the following figure.

![original synthetic data](./docs/images/original_time_series.png)
![sifted synthetic data](./docs/images/sifted_time_series.png)


## For developers

Run test cases with the following commands.

```shell
# Install dependencies for development
python -m pip install -r requirements-dev.txt
# Run test cases
pytest -s -v tests
```

## License

[BSD-3-Clause](LICENSE)
