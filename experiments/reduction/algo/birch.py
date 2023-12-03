
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import Birch as SKBirch

from .base import NormalityReducer


class BirchAD:
    """Anomaly Detector with Birch Clustering
    see paper [Gulenko+,CLOUD2018] "Detecting Anomalous Behavior of Black-Box Services Modeled with Distance-Based Online Clustering"
    """

    def __init__(self, threshold: float = 0.5, branching_factor: int = 50) -> None:
        self.birch = SKBirch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)

    def train(self, data: npt.NDArray) -> None:
        self.birch.fit(data)
        self.centroids_ = self.birch.subcluster_centers_
        self.radii_ = self.collect_radius()
        assert len(self.centroids_) == len(
            self.radii_
        ), f"len(self.centroids_)={len(self.centroids_)} != len(self.radii_)={len(self.radii_)}"

    def _get_leaves(self) -> list:
        leaf_ptr = self.birch.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    def collect_radius(self) -> np.ndarray:
        radii = []
        for leave in self._get_leaves():
            for subcluster in leave.subclusters_:
                radii.append(subcluster.radius)
        return np.array(radii)

    def is_normal(self, x: np.ndarray) -> bool:
        distances = np.array([np.linalg.norm(c - x) for c in self.centroids_])
        return bool(np.any(distances <= self.radii_))


class Birch(NormalityReducer):

    def run(self) -> pd.DataFrame:
        anomalous_start_idx: int = self.config["birch_anomalous_start_idx"]
        threshold: float = self.config.get("birch_threshold", 10.0)
        branching_factor: int = self.config.get("birch_branching_factor", 50)

        anomalous_data = self.data.iloc[anomalous_start_idx:, :]
        normal_data = self.data.iloc[anomalous_start_idx - anomalous_data.shape[0] : anomalous_start_idx, :]
        normal_mu, normal_sigma = normal_data.mean(), normal_data.std()

        def _zscore(x: npt.NDArray, mu: float, sigma: float) -> np.ndarray:
            if sigma == 0.0:
                sigma = 1
            return (x - mu) / sigma

        normal_data = normal_data.apply(lambda x: _zscore(x, normal_mu[x.name], normal_sigma[x.name]), axis=0)

        adtector = BirchAD(threshold=threshold, branching_factor=branching_factor)
        adtector.train(normal_data.values.T)

        remained_metrics = []
        for col in anomalous_data.columns:
            x = _zscore(anomalous_data[col].to_numpy(), normal_mu[col], normal_sigma[col])
            if not adtector.is_normal(x):
                remained_metrics.append(col)
        return self.data.loc[:, remained_metrics]

    def detect_anomaly(self, x: npt.NDArray) -> bool:
        raise NotImplementedError
