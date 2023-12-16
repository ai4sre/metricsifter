import warnings

import pandas as pd


def run_rcd(data_df: pd.DataFrame, boundary_index: int, top_k: int, n_iters: int) -> list[tuple[str, float]]:
    dataset = data_df.apply(zscore).dropna(how="any", axis=1)
    normal_data_df = dataset[dataset.index < boundary_index]
    abnormal_data_df = dataset[dataset.index >= boundary_index]

    model = RCD(config=RCDConfig(k=top_k, localized=True, start_alpha=0.001, ci_test=gsq))

    def _run_rcd() -> list[dict]:
        with threadpool_limits(limits=1):
            with warnings.catch_warnings(action='ignore', category=FutureWarning):
                return model.find_root_causes(normal_data_df, abnormal_data_df).to_list()

    if n_iters <= 1:
        return [(r["root_cause"], r["score"]) for r in _run_rcd()]
    # seed ensamble
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(_run_rcd)() for _ in range(n_iters))
    assert results is not None, "The results of rcd.rca_with_rcd are not empty"

    scores: dict[str, int] = defaultdict(int)
    for result in results:
        if result is None:
            continue
        for m in result[:top_k]:
            scores[m["root_cause"]] += 1
    return sorted([(metric, n / n_iters) for (metric, n) in scores.items()], key=lambda x: x[1], reverse=True)
