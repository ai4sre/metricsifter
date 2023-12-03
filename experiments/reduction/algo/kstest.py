import numpy as np
import numpy.typing as npt
import scipy.stats

from .base import NormalityReducer


class KSTest(NormalityReducer):

    def detect_anomaly(self, x: npt.NDArray, **kwargs: dict) -> bool:
        kstest_anomalous_start_idx: int = kwargs["kstest_anomalous_start_idx"]
        kstest_alpha: float = kwargs.get("kstest_alpha", 0.05)
        if kstest_anomalous_start_idx == 0:
            train_x, test_x = np.split(x, 2)
        else:
            train_x, test_x = np.split(x, [kstest_anomalous_start_idx])
        pval = scipy.stats.ks_2samp(train_x, test_x).pvalue
        return pval < kstest_alpha
