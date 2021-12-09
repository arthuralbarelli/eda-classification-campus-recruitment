import numpy as np
import pandas as pd
import pytest
from utility import load_data


def test_load_data():
    X, y, cols = load_data("data/Placement_Data_Full_Class.csv")

    # Type check
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.core.series.Series)
    assert isinstance(cols, list)

    # Shape check
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] >= 1
