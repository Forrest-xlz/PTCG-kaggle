"""Deterministic date-based expansion of training sample indices."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as calendar_date
from typing import Mapping, Sequence

import numpy as np


ReplayDate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DateSamplingCurve:
    mode: str
    start: float
    end: float
    exponent: float | None = None


@dataclass(frozen=True, slots=True)
class DateSamplingDateStats:
    date: ReplayDate
    coordinate: float
    weight: float
    source_samples: int
    weighted_samples: int


@dataclass(frozen=True, slots=True)
class DateSamplingResult:
    indices: np.ndarray
    dates: tuple[DateSamplingDateStats, ...]


def _ordinal(value: ReplayDate) -> int:
    month, day = map(int, value)
    return calendar_date(2001, month, day).toordinal()


def date_weights(
    dates: Sequence[ReplayDate],
    curve: DateSamplingCurve,
) -> dict[ReplayDate, float]:
    unique = sorted({(int(month), int(day)) for month, day in dates}, key=_ordinal)
    if not unique:
        raise ValueError("date sampling requires at least one training date")
    if curve.mode not in {"linear", "power"}:
        raise ValueError("date sampling mode must be linear or power")
    start, end = float(curve.start), float(curve.end)
    if not math.isfinite(start) or not math.isfinite(end) or min(start, end) < 0:
        raise ValueError("date sampling weights must be finite and non-negative")
    exponent = 1.0
    if curve.mode == "power":
        exponent = float(curve.exponent) if curve.exponent is not None else math.nan
        if not math.isfinite(exponent) or exponent <= 0:
            raise ValueError("date sampling exponent must be finite and positive")
    if len(unique) == 1:
        return {unique[0]: end}
    ordinals = [_ordinal(value) for value in unique]
    span = ordinals[-1] - ordinals[0]
    return {
        value: start + (end - start) * (((ordinal - ordinals[0]) / span) ** exponent)
        for value, ordinal in zip(unique, ordinals)
    }


def build_date_weighted_indices(
    indices: np.ndarray,
    dates: np.ndarray,
    *,
    weights: Mapping[ReplayDate, float],
    seed: int,
    enabled: bool = True,
) -> DateSamplingResult:
    if not enabled:
        return DateSamplingResult(indices=indices, dates=())
    indices = np.asarray(indices)
    dates = np.asarray(dates)
    if indices.ndim != 1 or dates.shape != (indices.size, 2):
        raise ValueError("indices and dates must have aligned one-dimensional rows")
    rng = np.random.default_rng(seed)
    expanded: list[np.ndarray] = []
    stats: list[DateSamplingDateStats] = []
    ordered_dates = sorted(weights, key=_ordinal)
    first, last = _ordinal(ordered_dates[0]), _ordinal(ordered_dates[-1])
    span = last - first
    for replay_date in ordered_dates:
        weight = float(weights[replay_date])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("date sampling weights must be finite and non-negative")
        mask = (dates[:, 0] == replay_date[0]) & (dates[:, 1] == replay_date[1])
        source = indices[mask]
        full_copies = math.floor(weight)
        pieces = [source] * full_copies
        fraction = weight - full_copies
        if fraction:
            pieces.append(source[rng.random(source.size) < fraction])
        selected = (
            np.concatenate(pieces).astype(indices.dtype, copy=False)
            if pieces
            else np.empty(0, dtype=indices.dtype)
        )
        expanded.append(selected)
        coordinate = 1.0 if span == 0 else (_ordinal(replay_date) - first) / span
        stats.append(
            DateSamplingDateStats(
                date=replay_date,
                coordinate=coordinate,
                weight=weight,
                source_samples=int(source.size),
                weighted_samples=int(selected.size),
            )
        )
    result = np.concatenate(expanded).astype(indices.dtype, copy=False)
    if result.size == 0:
        raise ValueError("date sampling produced zero training samples")
    return DateSamplingResult(indices=result, dates=tuple(stats))
