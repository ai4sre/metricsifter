import json
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Segment:
    """Information about a high-density period (segment)

    Attributes:
        label: Segment ID (sequential number starting from 0)
        start_time: Start time of the segment (minimum value of change points, row index)
        end_time: End time of the segment (maximum value of change points, row index)
    """

    label: int
    start_time: int
    end_time: int


@dataclass
class SegmentCandidate:
    """Lightweight view of one candidate segment, passed to a custom selection strategy.

    A custom ``segment_selection_method`` is any
    ``Callable[[SegmentCandidate], float]``: it receives one of these per
    candidate segment and returns a score; :class:`metricsifter.sifter.Sifter`
    keeps the segment with the maximum score (ties resolved by discovery order,
    matching the built-in ``"max"`` / ``"weighted_max"`` strategies).

    Attributes:
        label: Segment ID (the KDE cluster label).
        metrics: Metric names whose change points fall inside this segment.
        change_points: Sorted row positions of the change points in this segment.
        metric_to_cps: For each metric in ``metrics``, its full list of change
            points across the whole series (not just this segment). Enables
            weighting a metric by how "focused" its activity is, e.g. the built-in
            ``"weighted_max"`` scores ``sum(1 / len(metric_to_cps[m]))``.
    """

    label: int
    metrics: frozenset[str]
    change_points: list[int]
    metric_to_cps: dict[str, list[int]]


@dataclass(frozen=True)
class SegmentInfo:
    """Rich, diagnostic information about a single change-point cluster (segment).

    Unlike :class:`Segment`, positions and wall-clock timestamps are kept in
    separate fields so that positional (``RangeIndex``) and timestamped
    (``DatetimeIndex``) inputs are both represented losslessly.

    Attributes:
        label: Segment ID (sequential number starting from 0).
        metrics: Metrics whose change points fall inside this segment.
        start_index: Row position of the earliest change point in the segment.
        end_index: Row position of the latest change point in the segment.
        start_time: Wall-clock time of ``start_index`` (only for DatetimeIndex input).
        end_time: Wall-clock time of ``end_index`` (only for DatetimeIndex input).
        score: Segment-selection score exposed from the internal selection method
            (e.g. ``weighted_max``). The selected segment holds the maximum score.
        selected: Whether this segment was chosen as the densest one.
    """

    label: int
    metrics: frozenset[str]
    start_index: int
    end_index: int
    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None
    score: float = 0.0
    selected: bool = False

    def to_dict(self) -> dict:
        return {
            "label": int(self.label),
            "metrics": sorted(self.metrics),
            "start_index": int(self.start_index),
            "end_index": int(self.end_index),
            "start_time": self.start_time.isoformat() if self.start_time is not None else None,
            "end_time": self.end_time.isoformat() if self.end_time is not None else None,
            "score": float(self.score),
            "selected": bool(self.selected),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentInfo":
        return cls(
            label=d["label"],
            metrics=frozenset(d["metrics"]),
            start_index=d["start_index"],
            end_index=d["end_index"],
            start_time=pd.Timestamp(d["start_time"]) if d.get("start_time") is not None else None,
            end_time=pd.Timestamp(d["end_time"]) if d.get("end_time") is not None else None,
            score=d.get("score", 0.0),
            selected=d.get("selected", False),
        )


@dataclass(frozen=True)
class PenaltyTuning:
    """Diagnostic report of the ``penalty_adjust="auto"`` plateau search.

    The tuner sweeps ``penalty_adjust`` over a geometric ``grid`` and keeps, for
    each grid point, the change points detected across all metrics. Adjacent
    grid points are compared with a position-tolerant Jaccard similarity; the
    widest run of similar results (a *plateau*) marks penalties that are far
    from both the over-segmentation and the missed-detection regimes, and its
    midpoint is chosen as ``resolved``.

    Attributes:
        requested: The constructor value that triggered tuning (``"auto"``).
        resolved: The concrete ``penalty_adjust`` the pipeline actually used.
        grid: Candidate multipliers, in ascending order.
        n_change_points: Total change points detected at each grid point.
        adjacent_jaccard: Tolerant Jaccard similarity between consecutive grid
            points (length ``len(grid) - 1``).
        plateau: ``(low, high)`` grid values bounding the selected plateau, or
            ``None`` when no plateau was found and the default was used.
        reason: Why ``resolved`` was chosen (``"plateau"``, ``"no_plateau"``,
            ``"no_metrics"``).
    """

    requested: float | str
    resolved: float
    grid: list[float] = field(default_factory=list)
    n_change_points: list[int] = field(default_factory=list)
    adjacent_jaccard: list[float] = field(default_factory=list)
    plateau: tuple[float, float] | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "resolved": float(self.resolved),
            "grid": [float(a) for a in self.grid],
            "n_change_points": [int(n) for n in self.n_change_points],
            "adjacent_jaccard": [float(j) for j in self.adjacent_jaccard],
            "plateau": [float(self.plateau[0]), float(self.plateau[1])] if self.plateau is not None else None,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PenaltyTuning":
        plateau = d.get("plateau")
        return cls(
            requested=d["requested"],
            resolved=d["resolved"],
            grid=list(d.get("grid", [])),
            n_change_points=list(d.get("n_change_points", [])),
            adjacent_jaccard=list(d.get("adjacent_jaccard", [])),
            plateau=(plateau[0], plateau[1]) if plateau is not None else None,
            reason=d.get("reason", ""),
        )


@dataclass(frozen=True)
class BandwidthTuning:
    """Diagnostic report of the ``bandwidth="auto"`` stability selection.

    For each candidate bandwidth the tuner re-segments bootstrap resamples of
    the metrics (drawn with replacement, shared across candidates) and scores
    how consistently the same ``selected_metrics`` come out (mean pairwise
    Jaccard). Only candidates whose full-data segmentation yields at least two
    segments are admissible -- this blocks the degenerate "one giant segment"
    solution whose stability is trivially perfect.

    Attributes:
        requested: The constructor value that triggered tuning (``"auto"``).
        resolved: The concrete bandwidth the pipeline actually used.
        grid: Candidate bandwidths, in ascending order.
        stability: Mean pairwise Jaccard per candidate; ``None`` marks an
            inadmissible candidate (fewer than two full-data segments).
        n_segments: Full-data segment count per candidate.
        reason: Why ``resolved`` was chosen (``"stability"``, ``"unimodal"``,
            ``"too_few_change_points"``, ``"no_change_points"``).
    """

    requested: float | str
    resolved: float
    grid: list[float] = field(default_factory=list)
    stability: list[float | None] = field(default_factory=list)
    n_segments: list[int] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "resolved": float(self.resolved),
            "grid": [float(h) for h in self.grid],
            "stability": [float(s) if s is not None else None for s in self.stability],
            "n_segments": [int(n) for n in self.n_segments],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BandwidthTuning":
        return cls(
            requested=d["requested"],
            resolved=d["resolved"],
            grid=list(d.get("grid", [])),
            stability=list(d.get("stability", [])),
            n_segments=list(d.get("n_segments", [])),
            reason=d.get("reason", ""),
        )


@dataclass
class SiftResult:
    """Diagnostic, explainable result of :meth:`metricsifter.sifter.Sifter.sift`.

    Besides the filtered ``data``, it records *why* every input metric was kept
    or dropped, the change points detected per metric, and every candidate
    segment with its selection score. It is designed to be handed to an LLM
    agent or an MCP tool via :meth:`to_json` (the raw DataFrame is intentionally
    excluded from serialization -- only metric names, times and reasons).

    Exclusion reasons are mutually exclusive; together with ``selected_metrics``
    they partition the input columns.

    Attributes:
        data: Filtered DataFrame containing only the selected metrics
            (``None`` when reconstructed from :meth:`from_dict`).
        selected_metrics: Metrics retained in the densest segment.
        filtered_no_change: Metrics dropped by the simple no-variation filter.
        filtered_no_change_points: Metrics that passed the filter but had no
            change point detected.
        filtered_out_of_segment: Metrics with change points that fell outside the
            selected (densest) segment.
        metric_to_change_points: Per-metric change points as row positions.
        metric_to_change_times: Per-metric change points as wall-clock times
            (``None`` unless the input had a ``DatetimeIndex``).
        segments: Every candidate segment discovered during KDE segmentation.
        selected_segment: The chosen densest segment (``None`` when no change
            points were detected at all).
        penalty_tuning: Report of the ``penalty_adjust="auto"`` search
            (``None`` unless auto-tuning was requested).
        bandwidth_tuning: Report of the ``bandwidth="auto"`` search (``None``
            unless auto-tuning was requested).
    """

    data: pd.DataFrame | None
    selected_metrics: frozenset[str]
    filtered_no_change: frozenset[str]
    filtered_no_change_points: frozenset[str]
    filtered_out_of_segment: frozenset[str]
    metric_to_change_points: dict[str, list[int]] = field(default_factory=dict)
    metric_to_change_times: dict[str, list[pd.Timestamp]] | None = None
    segments: list[SegmentInfo] = field(default_factory=list)
    selected_segment: SegmentInfo | None = None
    penalty_tuning: PenaltyTuning | None = None
    bandwidth_tuning: BandwidthTuning | None = None

    def to_dict(self) -> dict:
        """Serialize to a plain, JSON-compatible dict (excludes the DataFrame)."""
        return {
            "selected_metrics": sorted(self.selected_metrics),
            "excluded": {
                "no_change_filter": sorted(self.filtered_no_change),
                "no_change_points": sorted(self.filtered_no_change_points),
                "out_of_segment": sorted(self.filtered_out_of_segment),
            },
            "metric_to_change_points": {
                metric: [int(cp) for cp in cps] for metric, cps in self.metric_to_change_points.items()
            },
            "metric_to_change_times": (
                {metric: [t.isoformat() for t in times] for metric, times in self.metric_to_change_times.items()}
                if self.metric_to_change_times is not None
                else None
            ),
            "segments": [segment.to_dict() for segment in self.segments],
            "selected_segment": self.selected_segment.to_dict() if self.selected_segment is not None else None,
            "penalty_tuning": self.penalty_tuning.to_dict() if self.penalty_tuning is not None else None,
            "bandwidth_tuning": self.bandwidth_tuning.to_dict() if self.bandwidth_tuning is not None else None,
        }

    def to_json(self, **kwargs) -> str:
        """Serialize to a JSON string. Extra kwargs are forwarded to ``json.dumps``."""
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "SiftResult":
        """Reconstruct from :meth:`to_dict` output. ``data`` is not restored (``None``)."""
        excluded = d["excluded"]
        raw_times = d.get("metric_to_change_times")
        metric_to_change_times = (
            {metric: [pd.Timestamp(t) for t in times] for metric, times in raw_times.items()}
            if raw_times is not None
            else None
        )
        selected = d.get("selected_segment")
        penalty_tuning = d.get("penalty_tuning")
        bandwidth_tuning = d.get("bandwidth_tuning")
        return cls(
            data=None,
            selected_metrics=frozenset(d["selected_metrics"]),
            filtered_no_change=frozenset(excluded["no_change_filter"]),
            filtered_no_change_points=frozenset(excluded["no_change_points"]),
            filtered_out_of_segment=frozenset(excluded["out_of_segment"]),
            metric_to_change_points={metric: list(cps) for metric, cps in d["metric_to_change_points"].items()},
            metric_to_change_times=metric_to_change_times,
            segments=[SegmentInfo.from_dict(s) for s in d["segments"]],
            selected_segment=SegmentInfo.from_dict(selected) if selected is not None else None,
            penalty_tuning=PenaltyTuning.from_dict(penalty_tuning) if penalty_tuning is not None else None,
            bandwidth_tuning=BandwidthTuning.from_dict(bandwidth_tuning) if bandwidth_tuning is not None else None,
        )

    @classmethod
    def from_json(cls, s: str) -> "SiftResult":
        """Reconstruct from a :meth:`to_json` string."""
        return cls.from_dict(json.loads(s))
