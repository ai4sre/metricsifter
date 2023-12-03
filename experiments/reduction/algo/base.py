from abc import ABC, abstractmethod

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd


class Reducer(ABC):
    def __init__(self, data: pd.DataFrame, n_jobs: int = 1, **kwargs: str | int | float):
        self.data = data
        self.n_jobs = n_jobs
        self.config = kwargs

    def run(self) -> pd.DataFrame:
        raise NotImplementedError

class NormalityReducer(Reducer):
    def __init__(self, data: pd.DataFrame, n_jobs: int = 1, **kwargs: str | int | float):
        super().__init__(data, n_jobs, **kwargs)

    @abstractmethod
    def detect_anomaly(self, x: npt.NDArray, **kwargs: dict[str, str | int | float]) -> bool:
        raise NotImplementedError

    def run(self) -> pd.DataFrame:
        ret = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self.detect_anomaly)(self.data[col].to_numpy(), **self.config) for col in self.data.columns
        )
        remained_metrics = [col for col, anomaly in zip(self.data.columns, ret) if anomaly]
        return self.data.loc[:, remained_metrics]


class RedundancyReducer(Reducer):
    def __init__(self, data: pd.DataFrame, n_jobs: int = 1, **kwargs: str | int | float):
        super().__init__(data, n_jobs, **kwargs)

    @abstractmethod
    def learn_clusters(self, dist_type: str) -> tuple[npt.NDArray, npt.NDArray]:
        raise NotImplementedError

    def run(self) -> pd.DataFrame:
        clusters, dist_matrix = self.learn_clusters(dist_type=str(self.config["dist_type"]))
        remained_metrics = self.select_representative_metrics(clusters, dist_matrix)
        return self.data.loc[:, remained_metrics]

    def select_representative_metrics(self, clusters: npt.NDArray, dist_matrix: npt.NDArray) -> list[str]:
        medoids = []
        unique_clusters = set(clusters)

        for cluster in unique_clusters:
            if cluster == -1:  # ignore noise points
                continue
            indices_in_cluster = np.where(clusters == cluster)[0]
            cluster_distance_matrix = dist_matrix[np.ix_(indices_in_cluster, indices_in_cluster)]
            medoid_index = np.argmin(cluster_distance_matrix.sum(axis=1))
            medoids.append(self.data.columns[medoid_index])

        return medoids
