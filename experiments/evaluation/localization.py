import numpy.typing as npt
import pandas as pd


def ac_k(X: pd.Series) -> pd.Series:
    k_values = list(range(1, X["k"].max()+1))
    acc_at_k = {k: .0 for k in k_values}
    num_anomalies: int = X["trial_no"].max()

    for k in k_values:
        for i in range(1, num_anomalies+1):
            x = X.loc[X["trial_no"] == i]
            num_root_causes = x["num_root_causes"].max()  # assume the same for all trials
            num_hit = x.loc[x["k"] <= k]["hit"].sum()
            acc_at_k[k] += num_hit / num_root_causes

        acc_at_k[k] /= num_anomalies

    return pd.Series(acc_at_k)


def avg_k(X: pd.Series) -> pd.Series:
    max_k = X["k"].max()
    acc = ac_k(X)
    return pd.Series({k: acc.iloc[:k].sum() / k for k in range(1, max_k + 1)})


def score(row: pd.Series) -> pd.Series:
    rec = ac_k(row)
    avgrec = avg_k(row)
    return pd.Series(
        {
            "ac@5": rec.get(5, .0), "avg@5": avgrec.get(5, .0),
            "elapsed_time_sum": row["elapsed_time_sum"].mean(),
            "elapsed_time_red": row["elapsed_time_red"].mean(),
            "elapsed_time_loc": row["elapsed_time_loc"].mean(),
            "ba": row["balanced_accuracy"].mean(),
            "recall": row["recall"].mean(),
            "specificity": row["specificity"].mean(),
        }
    )


def rs_score(row: pd.Series) -> pd.Series:
    rs_score = get_scores_of_random_selection(
        num_metrics=row["num_remained"].to_numpy(), num_found_metrics=(row["num_root_causes"]*row["root_cause_recall"]).to_numpy(),
        max_k=row["k"].max(),
    )
    return pd.Series(
        {
            "ac@5": rs_score.get("AC_5", .0), "avg@5": rs_score.get("AVG_5", .0),
            "elapsed_time_sum": row["elapsed_time_sum"].mean(),
            "elapsed_time_red": row["elapsed_time_red"].mean(),
            "elapsed_time_loc": row["elapsed_time_loc"].mean(),
            "ba": row["balanced_accuracy"].mean(),
            "recall": row["recall"].mean(),
            "specificity": row["specificity"].mean(),
        },
    )

def get_scores_of_random_selection(
    num_metrics: npt.NDArray, num_found_metrics: npt.NDArray, max_k: int = 5,
    lower_case: bool = False,
) -> dict[str, float]:

    def ac_k(n: npt.NDArray, g: npt.NDArray, k: int) -> float:
        prob_single_correct = g / n
        prob_at_least_one_correct = 1 - (1 - prob_single_correct) ** k
        return prob_at_least_one_correct.mean()

    def avg_k(ac_k_: dict) -> dict:
        return {k: sum([ac_k_[j] for j in range(1, k + 1)]) / k for k in range(1, max_k + 1)}

    ac_k_ = {k: ac_k(num_metrics, num_found_metrics, k) for k in range(1, max_k + 1)}
    avg_k_ = avg_k(ac_k_)

    ac_prefix = "ac_" if lower_case else "AC_"
    avg_prefix = "avg_" if lower_case else "AVG_"
    return dict({f"{ac_prefix}{k}": v for k, v in ac_k_.items()}, **{f"{avg_prefix}{k}": v for k, v in avg_k_.items()})
