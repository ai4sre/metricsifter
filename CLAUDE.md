# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Version Support

- **Core Package**: Python 3.10, 3.11, 3.12, 3.13, 3.14
- **Development/Testing**: Python 3.10, 3.11 only (due to sfr-pyrca dependency constraints)
- **Experiments**: Python 3.10, 3.11 only (requires sfr-pyrca)

## Development Commands

### Setup Development Environment
**Note**: Use Python 3.10 or 3.11 for development (uv automatically selects 3.11).

```bash
# Using uv (recommended for faster installation)
uv sync --all-extras

# Or using pip with Python 3.10 or 3.11
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest -s -v tests

# Run tests with verbose output (as used in CI)
pytest -s -vv tests
```

### Code Quality
```bash
# Format code with black
black .

# Lint code with ruff
ruff check .

# Type checking with mypy
mypy metricsifter/
```

### Package Installation
```bash
# Install package in development mode (using uv)
uv pip install -e .

# Install package in development mode (using pip)
pip install -e .

# Install from PyPI (using uv)
uv pip install metricsifter

# Install from PyPI (using pip)
pip install metricsifter
```

### Running Experiments
```bash
# Setup experiment environment with uv
uv sync --extra experiments

# Or using pip
cd experiments/
pip install -r requirements.txt

# Download datasets from GitHub releases
gh release download data-v1.0.0 -D data/ --pattern "*.tar.bz2"

# Run individual experiment scripts (examples from research paper)
# Reduction experiment
python -m experiments.reduction.runner

# Localization experiment
python -m experiments.localization.pyrca

# Evaluation with parameter sweeping
python -m experiments.sweeper.sweeper
```

## Architecture Overview

MetricSifter is a feature reduction framework for multivariate time series data that identifies anomalous metrics for fault localization in cloud applications.

### Core Components

#### Main Package (`metricsifter/`)
The core library implements a 3-phase pipeline:

**Phase 1: Simple Filtering** (`sifter.py:_filter_no_changes`)
- Removes metrics with constant values or no variation
- Uses parallel processing (configurable via `n_jobs`)
- Fast pre-filtering to reduce computational load

**Phase 2: Change Point Detection** (`algo/detection.py`)
- Detects change points per metric using ruptures library
- Supports 3 algorithms via `search_method` parameter:
  - `"pelt"` (default): PELT algorithm with kernel="linear" - fastest, suitable for most cases
  - `"binseg"`: Binary Segmentation - moderate speed, good for multiple change points
  - `"bottomup"`: Bottom-Up - slowest, most accurate for complex patterns
- Penalty calculation controlled by `penalty` ("aic"|"bic"|float) and `penalty_adjust` (multiplier)
- Handles missing values by treating NaN boundaries as change points

**Phase 3: Segmentation & Selection** (`algo/segmentation.py`)
- Groups change points using Kernel Density Estimation (KDE)
- `bandwidth` parameter controls clustering sensitivity
- Identifies segments (time windows) with high change point density
- Two selection strategies via `segment_selection_method`:
  - `"max"`: Select segment with most metrics
  - `"weighted_max"`: Weight by inverse of change point count per metric (prefers metrics with fewer, more significant changes)

**Parallel Processing** (`utils.py`)
- Wraps pandas operations for joblib parallelization
- Used in filtering and change point detection phases

#### Experiments Framework (`experiments/`)
Research evaluation pipeline for comparing reduction and localization methods:

**Data Pipeline** (`dataset/`)
- `loader.py`: Load time series datasets from files
- `separater.py`: Separate metrics by component (service/container granularity)
- `metric.py`: Metric metadata and ground truth handling

**Reduction Algorithms** (`reduction/`)
- `runner.py`: Unified interface for 8+ reduction methods
- `algo/`: Baseline implementations (NSigma, Birch, KSTest, FluxInfer-AD, HDBS)
- Supports component-level reduction for empirical datasets

**Fault Localization** (`localization/`)
- `pyrca.py`: Integration with PyRCA library (PC, LiNGAM, HT, PageRank, EpsilonDiagnosis)
- `rcd.py`: Random Causal Discovery implementation
- Uses call graph as prior knowledge to constrain causal discovery

**Evaluation Metrics** (`evaluation/`)
- `reduction.py`: Recall, specificity, balanced accuracy for reduction quality
- `localization.py`: AC@K, AVG@K metrics for top-K fault localization accuracy
- `empirical_ground_truth.py`: Ground truth validation for real-world datasets

**Prior Knowledge** (`priorknowledge/`)
- `call_graph.py`: Service dependency graphs for causal constraint
- `sockshop.py`, `trainticket.py`: Domain-specific knowledge for benchmark systems

**Parameter Sweeping** (`sweeper/`)
- Grid search framework for hyperparameter tuning

### Key Configuration Parameters
- `penalty_adjust`: Controls change point detection sensitivity (default: 2.0, higher = fewer change points)
- `bandwidth`: KDE bandwidth for change point clustering (default: 2.5, higher = wider segments)
- `segment_selection_method`: Segment selection strategy
  - `"max"`: Select segment with most metrics
  - `"weighted_max"`: Weight by 1/num_change_points per metric (default, better for precision)
- `search_method`: Change point detection algorithm ("pelt"|"binseg"|"bottomup")
- `cost_model`: Cost function for change point detection ("l2" default for BinSeg/BottomUp)
- `penalty`: Penalty type ("aic"|"bic"|float, default "bic")
- `n_jobs`: Parallel processing workers (default: 1, set to -1 for all CPUs)

### Two-Stage Pipeline Pattern
The experiments demonstrate a common pattern of combining reduction and localization:

```python
# Stage 1: Reduction (narrow down suspicious metrics)
sifter = Sifter(penalty_adjust=2.5, bandwidth=3.5)
reduced_data = sifter.run(full_data)

# Stage 2: Localization (identify root cause from reduced set)
from experiments.localization.pyrca import run_rca
results = run_rca(
    pk=prior_knowledge,
    data_df=reduced_data,
    building_step="PC",  # Causal discovery: PC or LiNGAM
    scoring_step="HT",   # Root cause ranking: HT or PageRank
    boundary_index=failure_start_time
)
```

Key insight: The reduction stage filters out normal/propagated metrics, making causal discovery more accurate and efficient.

### Dependencies
- **Core**: numpy, pandas, scipy, ruptures, statsmodels, networkx, joblib
- **Development**: black, ruff, pytest, coverage
- **Experiments**: scikit-learn, hdbscan, threadpoolctl, sfr-pyrca (PyRCA from Salesforce)

## Testing Requirements
- Supports Python 3.10 and 3.11
- CI runs on Ubuntu with C library dependencies (libc6-dev, gcc)
- Tests use synthetic data generation from `tests/sample_gen/generator.py`
- Test data simulates fault propagation in service dependency graphs
