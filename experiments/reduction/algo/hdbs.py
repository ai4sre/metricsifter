
import numpy as np
import numpy.typing as npt
import scipy.stats
from hdbscan import HDBSCAN
from numpy.fft import fft, ifft
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances

from .base import RedundancyReducer


def pearson(x: npt.NDArray, y: npt.NDArray) -> float:
    r = scipy.stats.pearsonr(x, y)[0]
    return 1 - abs(r) if r is not np.nan else 0.0


def sbd(x: npt.NDArray, y: npt.NDArray) -> float:
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    if dist < 0:
        return 0
    else:
        return dist


def _ncc_c(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    den = np.array(norm(x) * norm(y))
    den[den == 0] = np.Inf
    x_len = len(x)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len - 1) :], cc[:x_len]))
    return np.real(cc) / den


class HDBS(RedundancyReducer):

    def learn_clusters(self, dist_type: str) -> tuple[npt.NDArray, npt.NDArray]:
        match dist_type:
            case "pearson":
                dist_func = pearson
            case "sbd":
                dist_func = sbd
            case _:
                raise ValueError(f"Invalid distance type: {dist_type}")

        distance_matrix = pairwise_distances(self.data.T, metric=dist_func, force_all_finite=False)
        clusterer = HDBSCAN(
            min_cluster_size=2,
            metric="precomputed",
            allow_single_cluster=True,
            core_dist_n_jobs=1,
        ).fit(distance_matrix)
        return clusterer.labels_, distance_matrix
