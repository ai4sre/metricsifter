import numpy as np
import numpy.typing as npt

from .base import NormalityReducer


class NSigma(NormalityReducer):
    """Anomaly Detector with N-Sigma rule
    see paper [Lu+,CCGRID2022] "Generic and Robust Performance Diagnosis via Causal Inference for OLTP Database Systems"
    """

    def detect_anomaly(self, x: npt.NDArray, **kwargs: dict[str, int | float | str]) -> bool:
        anomalous_start_idx: int = kwargs["nsigma_anomalous_start_idx"]
        n_sigmas: float = kwargs.get("n_sigmas", 3.0)

        test_start_idx = x.shape[0] - (anomalous_start_idx + 1)
        train, test = x[:test_start_idx], x[test_start_idx:]

        mu, sigma = np.mean(train), np.std(train)
        if sigma == 0.0:
            sigma = 0.0001
        scores = np.abs((test - mu) / sigma)

        s_x: float = np.max(scores, axis=0)
        alpha_x: float = 0 if s_x < n_sigmas else np.log1p(s_x)
        return alpha_x > 0
