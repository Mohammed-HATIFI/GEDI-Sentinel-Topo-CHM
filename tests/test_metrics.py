import numpy as np
from gedi_sentinel_chm.metrics import regression_metrics


def test_perfect_prediction():
    result = regression_metrics([1, 2, 3], [1, 2, 3])
    assert result["n"] == 3
    assert result["mae"] == 0
    assert result["rmse"] == 0
    assert np.isclose(result["r2"], 1)
    assert np.isclose(result["slope"], 1)


def test_nonfinite_pairs_are_removed_together():
    result = regression_metrics([1, np.nan, 3], [1, 2, 5])
    assert result["n"] == 2
    assert np.isclose(result["mae"], 1)
