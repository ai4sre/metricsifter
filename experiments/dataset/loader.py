import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

DATA_DIR = (Path(__file__).parent.parent / "data").absolute().resolve()

SYNTHETIC_DATA_FILE = DATA_DIR / "synthetic_data.tar.bz2"

EMPIRICAL_SS_SMALL_DATA_FILE = DATA_DIR / "ss-small.tar.bz2"
EMPIRICAL_SS_MEDIUM_DATA_FILE = DATA_DIR / "ss-medium.tar.bz2"
EMPIRICAL_SS_LARGE_DATA_FILE = DATA_DIR / "ss-large.tar.bz2"
EMPIRICAL_TT_SMALL_DATA_FILE = DATA_DIR / "tt-small.tar.bz2"
EMPIRICAL_TT_MEDIUM_DATA_FILE = DATA_DIR / "tt-medium.tar.bz2"
EMPIRICAL_TT_LARGE_DATA_FILE = DATA_DIR / "tt-large.tar.bz2"

SOCKSHOP_DATA_FILES = (EMPIRICAL_SS_SMALL_DATA_FILE, EMPIRICAL_SS_MEDIUM_DATA_FILE, EMPIRICAL_SS_LARGE_DATA_FILE)
TRAINTICKET_DATA_FILES = (EMPIRICAL_TT_SMALL_DATA_FILE, EMPIRICAL_TT_MEDIUM_DATA_FILE, EMPIRICAL_TT_LARGE_DATA_FILE)

SYNTHETIC_PARAM_PATTERN = re.compile(r"anomaly_type-(?P<anomaly_type>\d+)_func_type-(?P<func_type>\w+)_noise_type-(?P<noise_type>\w+)_weight_generator-(?P<weight_generator>\w+)")


def _transform_dict_to_array(result: dict) -> list[tuple[dict, dict]]:
    return [  # data_params + data
        ({ "dataset_name": dataset_name,
          "anomaly_type": anomaly_type,
          "func_type": func_type, "noise_type": noise_type,
          "weight_generator": weight_generator,
          "trial_no": trial_no }
        , data)
        for (dataset_name, anomaly_type, func_type, noise_type, weight_generator, trial_no), data in result.items()
    ]


def load_synthetic_data() -> list[tuple[dict, dict]]:
    result: dict = defaultdict(lambda: defaultdict(dict))
    with tarfile.open(SYNTHETIC_DATA_FILE.as_posix(), "r:bz2") as tar:
        for tarinfo in tar:
            if not tarinfo.isfile():
                continue
            dataset_name = tarinfo.name.split("/")[1]
            path = Path(tarinfo.name)

            trial_no: int = int(path.parent.name)

            ret = SYNTHETIC_PARAM_PATTERN.match(path.parent.parent.name)
            if ret is None:
                raise ValueError(f"Unknown parameters pattern: {path.parent.name}")
            anomaly_type: int = int(ret.group("anomaly_type"))
            func_type: str = ret.group("func_type")
            noise_type: str = ret.group("noise_type")
            weight_generator: str = ret.group("weight_generator")
            params: tuple = (dataset_name, anomaly_type, func_type, noise_type, weight_generator, trial_no)

            f = tar.extractfile(tarinfo.name)
            if f is None:
                raise ValueError(f"{path.name} is None")
            match path.name:
                case "normal_data.csv":
                    normal_data = pd.read_csv(f)
                    result[params]["normal_data"] = normal_data
                case "abnormal_data.csv":
                    abnormal_data = pd.read_csv(f)
                    result[params]["abnormal_data"] = abnormal_data
                case "ground_truth.json":
                    ground_truth = json.load(f)
                    result[params]["ground_truth"] = ground_truth
                case "graph_adjacency.csv":
                    graph_adjacency = pd.read_csv(f)
                    result[params]["graph_adjacency"] = graph_adjacency
                case _:
                    raise ValueError(f"Unknown file: {path.name}")
    return _transform_dict_to_array(result)


def _load_empirical_tar_file(tarinfo: tarfile.TarInfo, tar_bytes: bytes) -> tuple[tuple[str, str, str, int], dict]:
    dataset_name, fault_type, fault_comp = tarinfo.name.split("/")[0:3]
    path = Path(tarinfo.name)
    trial_no: int = int(path.parent.name)

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:bz2') as tar:
        f = tar.extractfile(tarinfo.name)
        if f is None:
            raise ValueError(f"{path.name} is None")

        params = (dataset_name, fault_type, fault_comp, trial_no)
        match path.name:
            case "metrics.csv":
                data = pd.read_csv(f, index_col=0)
                return params, { "data": data }
            case "data_spec.json":
                data_spec = json.load(f)
                return params, { "data_spec": data_spec }
            case _:
                raise ValueError(f"Unknown file: {path.name}")


def _transform_dict_to_array_for_empirical(result: dict) -> list[tuple[dict, dict]]:
    return [
        ({
            "dataset_name": dataset_name,
            "fault_type": fault_type,
            "fault_comp": fault_comp,
            "trial_no": trial_no,
        }, data)
        for _, nested_items in result.items()
        for (dataset_name, fault_type, fault_comp, trial_no), data in nested_items.items()
    ]


def _load_empirical_data(data_paths: tuple[Path,...], n_jobs: int = -1) -> list[tuple[dict, dict]]:
    result: dict = defaultdict(lambda: defaultdict(dict))

    for data_path in data_paths:
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} does not exist")

        with open(data_path.as_posix(), 'rb') as f:
            tar_bytes = f.read()
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:bz2') as tar:
            members = [member for member in tar.getmembers() if member.isfile() ]

        ret = Parallel(n_jobs=n_jobs)(delayed(_load_empirical_tar_file)(member, tar_bytes) for member in members)
        assert ret is not None, "Parallel execution failed"
        for params, load_file in ret:
            result[params[0]][params].update(load_file)

    return _transform_dict_to_array_for_empirical(result)


def load_sockshop_data(n_jobs: int = -1) -> list[tuple[dict, dict]]:
    return _load_empirical_data(SOCKSHOP_DATA_FILES, n_jobs)


def load_trainticket_data(n_jobs: int = -1) -> list[tuple[dict, dict]]:
    return _load_empirical_data(TRAINTICKET_DATA_FILES, n_jobs)
