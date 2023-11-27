from typing import Any, Callable, Generator

import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs


def gen_even_slices(n: int, n_packs: int, *, n_samples: int | None=None) -> Generator[slice, None, None]:
    """Generator to create n_packs slices going up to n."""

    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1" % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples is not None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end


def parallel_apply(df: pd.DataFrame, func: Callable, n_jobs: int = -1, **kwargs: dict[str, Any]) -> pd.DataFrame:
    """Pandas apply in parallel using joblib."""

    if effective_n_jobs(n_jobs) == 1:
        return df.apply(func, **kwargs)
    else:
        ret = Parallel(n_jobs=n_jobs)(
            delayed(type(df).apply)(df.iloc[:, s], func, **kwargs)
            for s in gen_even_slices(df.size, effective_n_jobs(n_jobs))
        )
        return pd.concat([r for r in ret if not r.empty])
