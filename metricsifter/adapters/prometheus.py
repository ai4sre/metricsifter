"""Convert a Prometheus ``query_range`` response into a MetricSifter DataFrame.

This adapter is a **pure transform**: it takes the already-parsed JSON ``dict``
of a Prometheus HTTP API ``/api/v1/query_range`` response (``resultType ==
"matrix"``) and returns a wide :class:`pandas.DataFrame` ready for
:meth:`Sifter.sift`. It performs no HTTP itself.

Design of the column-name <-> label-set correspondence
------------------------------------------------------
Each Prometheus series is identified by a label set (a dict such as
``{"__name__": "node_cpu_seconds_total", "cpu": "0", "mode": "idle"}``). We turn
each label set into a **deterministic** column name in the canonical PromQL form
``name{k1="v1",k2="v2"}`` with labels sorted by key, so the same series always
maps to the same column regardless of dict ordering. Because that string form is
lossy to parse back reliably (label values may contain the separator), we do not
re-parse it. Instead the exact ``column -> label dict`` mapping is preserved as
structured metadata on the returned frame (``df.attrs["metric_labels"]``) and
exposed via :func:`to_metric_labels`. In the rare event two series collapse to
the same canonical name, a deterministic ``#<n>`` suffix disambiguates them.
"""

from __future__ import annotations

import pandas as pd

METRIC_LABELS_ATTR = "metric_labels"


def _canonical_column_name(metric: dict[str, str], label_sep: str) -> str:
    """Build a deterministic PromQL-style column name from a label set."""
    name = metric.get("__name__", "")
    labels = sorted((k, v) for k, v in metric.items() if k != "__name__")
    if not labels:
        return name
    body = label_sep.join(f'{k}="{v}"' for k, v in labels)
    return f"{name}{{{body}}}"


def from_query_range(response: dict, *, label_sep: str = ",") -> pd.DataFrame:
    """Convert a matrix ``query_range`` response to a wide DataFrame.

    Args:
        response: Parsed Prometheus response dict. Must contain
            ``data.result`` in matrix form (each item has ``metric`` and
            ``values``: a list of ``[unix_ts, "string_value"]`` pairs).
        label_sep: Separator used between labels inside the canonical column
            name (default ``","``).

    Returns:
        A wide ``pd.DataFrame`` with a UTC ``DatetimeIndex`` and one column per
        series. Values are coerced to ``float`` (Prometheus returns strings).
        Series with mismatched timestamps are aligned by an outer join, leaving
        ``NaN`` where a series has no sample (handled downstream by the sift
        NaN support). The ``column -> label dict`` mapping is stored on
        ``df.attrs["metric_labels"]``; use :func:`to_metric_labels` to read it.

    Raises:
        ValueError: If the response is not a matrix result.
    """
    data = response.get("data", {})
    result_type = data.get("resultType")
    if result_type is not None and result_type != "matrix":
        raise ValueError(f"from_query_range expects a matrix result (query_range), got resultType={result_type!r}.")
    result = data.get("result", [])

    series_list: list[pd.Series] = []
    column_to_labels: dict[str, dict[str, str]] = {}
    used_names: dict[str, int] = {}

    for item in result:
        metric: dict[str, str] = item.get("metric", {})
        values = item.get("values", [])

        base_name = _canonical_column_name(metric, label_sep)
        # Deterministically disambiguate accidental name collisions.
        if base_name in used_names:
            used_names[base_name] += 1
            column = f"{base_name}#{used_names[base_name]}"
        else:
            used_names[base_name] = 0
            column = base_name

        timestamps = pd.to_datetime([v[0] for v in values], unit="s", utc=True)
        floats = [float(v[1]) for v in values]
        series = pd.Series(floats, index=timestamps, name=column, dtype="float64")
        # Collapse any duplicated timestamps within a single series (keep last).
        series = series[~series.index.duplicated(keep="last")]

        series_list.append(series)
        column_to_labels[column] = dict(metric)

    if not series_list:
        df = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    else:
        # Outer join on the union of all timestamps; NaN where a series is absent.
        df = pd.concat(series_list, axis=1).sort_index()

    df.attrs[METRIC_LABELS_ATTR] = column_to_labels
    return df


def to_metric_labels(df: pd.DataFrame, column: str) -> dict[str, str]:
    """Reverse-lookup the original Prometheus label set for a column.

    Args:
        df: A DataFrame produced by :func:`from_query_range`.
        column: A column name in ``df``.

    Returns:
        The original label dict (including ``__name__``) for ``column``.

    Raises:
        KeyError: If the frame carries no adapter metadata or ``column`` is
            unknown.
    """
    mapping = df.attrs.get(METRIC_LABELS_ATTR)
    if mapping is None:
        raise KeyError(
            "This DataFrame has no Prometheus label metadata. " "It must be produced by prometheus.from_query_range()."
        )
    if column not in mapping:
        raise KeyError(f"Unknown column {column!r}. Known columns: {sorted(mapping)}.")
    return dict(mapping[column])
