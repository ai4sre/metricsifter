from typing import Any, Callable, Generator

import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs


def gen_even_slices(n: int, n_packs: int, *, n_samples: int | None = None) -> Generator[slice, None, None]:
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
    """Apply ``func`` column-wise in parallel using joblib.

    The columns of ``df`` are split into even packs (one pack per effective job)
    and each pack is processed by a worker. The per-pack results are concatenated
    back in column order, so the output is identical to ``df.apply(func)`` --
    only faster. Slicing is done on ``df.shape[1]`` (the number of columns), not
    ``df.size`` (rows * columns): the latter made the first pack swallow every
    column while the remaining workers received empty slices, silently disabling
    parallelism.
    """

    if effective_n_jobs(n_jobs) == 1 or df.shape[1] == 0 or df.shape[0] == 0:
        return df.apply(func, **kwargs)
    ret = Parallel(n_jobs=n_jobs)(
        delayed(type(df).apply)(df.iloc[:, s], func, **kwargs)
        for s in gen_even_slices(df.shape[1], effective_n_jobs(n_jobs))
    )
    results = [r for r in ret if not r.empty]
    if not results:
        return df.apply(func, **kwargs)
    if isinstance(results[0], pd.DataFrame):
        # func mapped each column to a same-length Series, so each worker
        # returns a column-subset DataFrame: stitch them back side by side.
        return pd.concat(results, axis=1)
    return pd.concat(results)
