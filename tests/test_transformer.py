"""Tests for the scikit-learn-compatible SifterTransformer.

The transformer must work with no scikit-learn present (duck typing), and
additionally integrate with scikit-learn's Pipeline/clone when it is installed.
"""

import pandas as pd
import pytest

from metricsifter import Sifter, SifterTransformer
from metricsifter.types import SiftResult
from tests.conftest import make_synthetic


class TestFitTransform:
    def test_fit_sets_learned_attributes(self):
        data = make_synthetic()
        tr = SifterTransformer(penalty_adjust=2.0, n_jobs=1).fit(data)

        assert isinstance(tr.result_, SiftResult)
        assert set(tr.selected_metrics_) == {"failure_0", "failure_1", "failure_2"}
        assert tr.n_features_in_ == data.shape[1]
        assert list(tr.feature_names_in_) == list(data.columns)

    def test_transform_same_data(self):
        data = make_synthetic()
        tr = SifterTransformer(penalty_adjust=2.0, n_jobs=1).fit(data)
        out = tr.transform(data)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == tr.selected_metrics_
        assert len(out) == len(data)

    def test_transform_new_data_with_same_schema(self):
        data = make_synthetic()
        tr = SifterTransformer(penalty_adjust=2.0, n_jobs=1).fit(data)
        # Different values, same columns: selection is fixed at fit time.
        new_data = make_synthetic()
        new_data.iloc[:, :] = new_data.to_numpy() + 100.0
        out = tr.transform(new_data)
        assert list(out.columns) == tr.selected_metrics_

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        data = make_synthetic()
        a = SifterTransformer(penalty_adjust=2.0, n_jobs=1).fit_transform(data)
        b = SifterTransformer(penalty_adjust=2.0, n_jobs=1).fit(data).transform(data)
        pd.testing.assert_frame_equal(a, b)

    def test_transform_missing_columns_raises_value_error(self):
        data = make_synthetic()
        tr = SifterTransformer(penalty_adjust=2.0, n_jobs=1).fit(data)
        broken = data.drop(columns=["failure_0"])
        with pytest.raises(ValueError, match="missing"):
            tr.transform(broken)

    def test_transform_before_fit_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            SifterTransformer().transform(make_synthetic())

    def test_non_dataframe_input_raises_value_error(self):
        with pytest.raises(ValueError, match="DataFrame"):
            SifterTransformer().fit(make_synthetic().to_numpy())


class TestParams:
    def test_get_params_matches_constructor(self):
        tr = SifterTransformer(penalty_adjust=3.0, bandwidth=1.0, n_jobs=2)
        params = tr.get_params()
        assert params["penalty_adjust"] == 3.0
        assert params["bandwidth"] == 1.0
        assert params["n_jobs"] == 2
        assert params["search_method"] == "pelt"

    def test_set_get_roundtrip(self):
        tr = SifterTransformer()
        tr.set_params(penalty_adjust=5.0, search_method="binseg")
        params = tr.get_params()
        assert params["penalty_adjust"] == 5.0
        assert params["search_method"] == "binseg"
        # A fresh instance built from get_params reproduces the same params.
        clone_like = SifterTransformer(**params)
        assert clone_like.get_params() == params

    def test_set_params_rejects_unknown(self):
        with pytest.raises(ValueError, match="Invalid parameter"):
            SifterTransformer().set_params(does_not_exist=1)

    def test_params_cover_all_sifter_args(self):
        # Every Sifter constructor argument must be a transformer param so the
        # wrapper stays faithful to the underlying estimator.
        import inspect

        sifter_args = set(inspect.signature(Sifter.__init__).parameters) - {"self"}
        tr_params = set(SifterTransformer().get_params())
        assert sifter_args <= tr_params


def _import_sklearn_or_skip(module: str):
    """Import a scikit-learn module, skipping only when it is unavailable."""
    return pytest.importorskip(module)


class TestSklearnIntegration:
    """Only runs when scikit-learn is installed and importable."""

    def test_clone_roundtrip(self):
        sklearn_base = _import_sklearn_or_skip("sklearn.base")
        tr = SifterTransformer(penalty_adjust=4.0, bandwidth=1.5, n_jobs=1)
        cloned = sklearn_base.clone(tr)
        assert cloned.get_params() == tr.get_params()
        assert cloned is not tr

    def test_pipeline_fit_transform(self):
        pipeline_mod = _import_sklearn_or_skip("sklearn.pipeline")
        data = make_synthetic()
        pipe = pipeline_mod.Pipeline([("sift", SifterTransformer(penalty_adjust=2.0, n_jobs=1))])
        out = pipe.fit_transform(data)
        assert list(out.columns) == ["failure_0", "failure_1", "failure_2"]
