import numpy as np
import pytest

from training.date_sampling import (
    DateSamplingCurve,
    build_date_weighted_indices,
    date_weights,
)


def test_linear_weights_use_elapsed_calendar_days() -> None:
    dates = [(7, 1), (7, 3), (7, 5)]
    weights = date_weights(dates, DateSamplingCurve("linear", 0.5, 1.5))

    assert weights == {(7, 1): 0.5, (7, 3): 1.0, (7, 5): 1.5}


def test_power_weights_and_single_date_endpoint() -> None:
    curve = DateSamplingCurve("power", 0.5, 1.5, exponent=2.0)

    assert date_weights([(7, 1), (7, 3), (7, 5)], curve) == {
        (7, 1): 0.5,
        (7, 3): 0.75,
        (7, 5): 1.5,
    }
    assert date_weights([(7, 5)], curve) == {(7, 5): 1.5}


def test_expansion_supports_zero_fractional_and_multiple_copies() -> None:
    indices = np.arange(12, dtype=np.uint32)
    dates = np.asarray(
        [[7, 1]] * 3 + [[7, 2]] * 3 + [[7, 3]] * 3 + [[7, 4]] * 3,
        dtype=np.int16,
    )
    result = build_date_weighted_indices(
        indices,
        dates,
        weights={(7, 1): 0.0, (7, 2): 0.5, (7, 3): 1.2, (7, 4): 2.5},
        seed=17,
    )
    counts = np.bincount(result.indices, minlength=len(indices))

    assert not counts[:3].any()
    assert np.all(counts[3:6] <= 1)
    assert np.all((1 <= counts[6:9]) & (counts[6:9] <= 2))
    assert np.all((2 <= counts[9:]) & (counts[9:] <= 3))
    assert result.indices.dtype == np.uint32


def test_fractional_sampling_is_reconstructed_by_seed() -> None:
    indices = np.arange(1000, dtype=np.uint32)
    dates = np.full((1000, 2), (7, 1), dtype=np.int16)
    kwargs = dict(indices=indices, dates=dates, weights={(7, 1): 0.5})

    first = build_date_weighted_indices(**kwargs, seed=42).indices
    second = build_date_weighted_indices(**kwargs, seed=42).indices
    different = build_date_weighted_indices(**kwargs, seed=43).indices

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_disabled_sampling_returns_original_indices() -> None:
    indices = np.arange(5, dtype=np.uint32)
    dates = np.full((5, 2), (7, 1), dtype=np.int16)

    result = build_date_weighted_indices(
        indices, dates, weights={(7, 1): 0.0}, seed=1, enabled=False
    )

    assert result.indices is indices
    assert result.dates == ()


def test_enabled_sampling_rejects_empty_result() -> None:
    with pytest.raises(ValueError, match="zero training samples"):
        build_date_weighted_indices(
            np.arange(2, dtype=np.uint32),
            np.full((2, 2), (7, 1), dtype=np.int16),
            weights={(7, 1): 0.0},
            seed=1,
        )
