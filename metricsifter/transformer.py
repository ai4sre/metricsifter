"""A scikit-learn-compatible transformer wrapper around :class:`Sifter`.

This module deliberately adds **no dependency on scikit-learn**. Compatibility is
achieved by duck-typing the estimator API (``fit`` / ``transform`` /
``fit_transform`` / ``get_params`` / ``set_params``) so that ``SifterTransformer``
drops into a :class:`sklearn.pipeline.Pipeline` and survives
:func:`sklearn.base.clone` when scikit-learn happens to be installed.

``clone`` compatibility is the load-bearing constraint: it requires that
``get_params`` returns exactly the constructor arguments and that ``__init__``
stores each one unmodified under the same name (the scikit-learn estimator
convention). We follow that convention strictly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metricsifter.sifter import Sifter
from metricsifter.types import SiftResult

# The constructor parameters shared with Sifter, plus the sift() switch. Kept in
# one place so get_params / set_params / clone stay in lock-step with __init__.
_PARAM_NAMES = (
    "search_method",
    "cost_model",
    "penalty",
    "penalty_adjust",
    "bandwidth",
    "segment_selection_method",
    "n_jobs",
    "without_simple_filter",
)


class SifterTransformer:
    """Select the failure-related metrics of a DataFrame, sklearn-style.

    ``fit`` runs the sift and remembers which columns to keep; ``transform``
    returns those columns from any DataFrame with a matching schema. Fitted
    attributes follow the scikit-learn trailing-underscore convention:

    - ``result_``: the full :class:`SiftResult` from the fit.
    - ``selected_metrics_``: selected column names, in the fitted input order.
    - ``n_features_in_`` / ``feature_names_in_``: the fitted input schema.
    """

    def __init__(
        self,
        search_method: str = "pelt",
        cost_model: str = "l2",
        penalty: str | float = "bic",
        penalty_adjust: float = 2.0,
        bandwidth: float = 2.5,
        segment_selection_method: str = "weighted_max",
        n_jobs: int = 1,
        without_simple_filter: bool = False,
    ) -> None:
        # Store every argument verbatim under its own name (sklearn convention;
        # required for get_params/clone round-trips to be exact).
        self.search_method = search_method
        self.cost_model = cost_model
        self.penalty = penalty
        self.penalty_adjust = penalty_adjust
        self.bandwidth = bandwidth
        self.segment_selection_method = segment_selection_method
        self.n_jobs = n_jobs
        self.without_simple_filter = without_simple_filter

    # -- scikit-learn estimator protocol ---------------------------------

    def get_params(self, deep: bool = True) -> dict:
        """Return constructor parameters (``deep`` accepted for API parity)."""
        return {name: getattr(self, name) for name in _PARAM_NAMES}

    def set_params(self, **params) -> "SifterTransformer":
        """Set constructor parameters in place and return ``self``."""
        for key, value in params.items():
            if key not in _PARAM_NAMES:
                raise ValueError(
                    f"Invalid parameter {key!r} for SifterTransformer. "
                    f"Valid parameters are: {sorted(_PARAM_NAMES)}."
                )
            setattr(self, key, value)
        return self

    def _build_sifter(self) -> Sifter:
        return Sifter(
            search_method=self.search_method,
            cost_model=self.cost_model,
            penalty=self.penalty,
            penalty_adjust=self.penalty_adjust,
            bandwidth=self.bandwidth,
            segment_selection_method=self.segment_selection_method,
            n_jobs=self.n_jobs,
        )

    @staticmethod
    def _check_dataframe(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"SifterTransformer requires a pandas DataFrame, got {type(X).__name__}. "
                "Column names are needed to select metrics."
            )
        return X

    def fit(self, X, y=None) -> "SifterTransformer":
        """Run the sift on ``X`` and remember the selected metrics.

        ``y`` is ignored (accepted for Pipeline compatibility).
        """
        X = self._check_dataframe(X)
        sifter = self._build_sifter()
        result: SiftResult = sifter.sift(X, without_simple_filter=self.without_simple_filter)

        self.result_ = result
        # Preserve the fitted input column order for a stable transform output.
        self.selected_metrics_ = [c for c in X.columns if c in result.selected_metrics]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> pd.DataFrame:
        """Return the fitted selected columns from ``X``.

        Raises a clear ``ValueError`` (never a bare ``KeyError``) when ``X`` is
        missing any selected column.
        """
        if not hasattr(self, "selected_metrics_"):
            raise ValueError("This SifterTransformer is not fitted yet. Call 'fit' before 'transform'.")
        X = self._check_dataframe(X)

        missing = [c for c in self.selected_metrics_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} column(s) selected during fit: {missing}. "
                "transform requires the same schema as fit."
            )
        return X.loc[:, self.selected_metrics_]

    def fit_transform(self, X, y=None) -> pd.DataFrame:
        """Convenience: ``fit(X).transform(X)``."""
        return self.fit(X, y).transform(X)
