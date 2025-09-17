# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_median_basic():
    """Test basic median functionality following jpivarski's example from issue #741."""
    array = ak.Array([[2.2, 1.1, 0.0], [], [3.3, 4.4], [5.5], [8.8, 6.6, 7.7, 9.9]])
    
    result = ak.median(array, axis=1)
    expected = [1.1, np.nan, 3.85, 5.5, 8.25]
    
    assert len(result) == len(expected)
    for i, (actual, exp) in enumerate(zip(result.tolist(), expected)):
        if np.isnan(exp):
            assert np.isnan(actual), f"Expected NaN at index {i}, got {actual}"
        else:
            assert actual == pytest.approx(exp), f"Mismatch at index {i}: {actual} != {exp}"


def test_median_axis_innermost():
    """Test median along innermost axis (axis=-1)."""
    array = ak.Array([[0, 1, 2, 3], [], [4, 5]])
    
    result = ak.median(array, axis=-1)
    expected = [1.5, np.nan, 4.5]
    
    assert len(result) == len(expected)
    for i, (actual, exp) in enumerate(zip(result.tolist(), expected)):
        if np.isnan(exp):
            assert np.isnan(actual), f"Expected NaN at index {i}, got {actual}"
        else:
            assert actual == pytest.approx(exp), f"Mismatch at index {i}: {actual} != {exp}"


def test_median_axis_none():
    """Test median of entire array (axis=None)."""
    array = ak.Array([[0, 1, 2, 3], [], [4, 5]])
    
    result = ak.median(array, axis=None)
    expected = 2.5  # median of [0, 1, 2, 3, 4, 5]
    
    assert result == pytest.approx(expected)


def test_median_axis_zero_regular():
    """Test median along axis=0 for regular arrays."""
    array = ak.Array([[0, 1, 2], [3, 4, 5]])
    
    result = ak.median(array, axis=0)
    expected = [1.5, 2.5, 3.5]  # median of [0,3], [1,4], [2,5]
    
    assert result.tolist() == pytest.approx(expected)


def test_median_axis_zero_irregular():
    """Test that axis=0 with irregular arrays raises appropriate error."""
    array = ak.Array([[0, 1, 2, 3], [], [4, 5]])
    
    with pytest.raises(NotImplementedError, match="Median along axis=0 for irregular arrays"):
        ak.median(array, axis=0)


def test_median_keepdims():
    """Test keepdims parameter."""
    array = ak.Array([[2.2, 1.1, 0.0], [3.3, 4.4], [5.5]])
    
    # Without keepdims
    result_no_keepdims = ak.median(array, axis=1, keepdims=False)
    assert result_no_keepdims.ndim == 1
    
    # With keepdims
    result_keepdims = ak.median(array, axis=1, keepdims=True)
    assert result_keepdims.ndim == 2
    assert result_keepdims.tolist() == [[1.1], [3.85], [5.5]]


def test_median_mask_identity():
    """Test mask_identity parameter."""
    array = ak.Array([[1, 2, 3], []])
    
    # mask_identity=False (default) - empty lists become NaN
    result_false = ak.median(array, axis=1, mask_identity=False)
    expected_false = [2.0, np.nan]
    for i, (actual, exp) in enumerate(zip(result_false.tolist(), expected_false)):
        if np.isnan(exp):
            assert np.isnan(actual), f"Expected NaN at index {i}, got {actual}"
        else:
            assert actual == pytest.approx(exp)
    
    # mask_identity=True - empty lists become None
    result_true = ak.median(array, axis=1, mask_identity=True)
    assert result_true.tolist() == [2.0, None]


def test_nanmedian_basic():
    """Test nanmedian functionality."""
    array = ak.Array([[2.2, 1.1, np.nan], [np.nan], [3.3, 4.4], [5.5]])
    
    # Regular median includes NaN in calculation
    result_median = ak.median(array, axis=1)
    
    # nanmedian ignores NaN values
    result_nanmedian = ak.nanmedian(array, axis=1)
    
    # First sublist: median of [2.2, 1.1] (ignoring NaN) = 1.65
    # Second sublist: empty after removing NaN, so None (mask_identity=True by default)
    # Third sublist: median of [3.3, 4.4] = 3.85
    # Fourth sublist: median of [5.5] = 5.5
    
    expected_nanmedian = [1.65, None, 3.85, 5.5]
    result_list = result_nanmedian.tolist()
    
    for i, (actual, exp) in enumerate(zip(result_list, expected_nanmedian)):
        if exp is None:
            assert actual is None, f"Expected None at index {i}, got {actual}"
        else:
            assert actual == pytest.approx(exp), f"Mismatch at index {i}: {actual} != {exp}"


def test_nanmedian_axis_none():
    """Test nanmedian with axis=None."""
    array = ak.Array([[1.0, np.nan, 3.0], [4.0, 5.0]])
    
    result = ak.nanmedian(array, axis=None)
    expected = 3.0  # median of [1.0, 3.0, 4.0, 5.0] (ignoring NaN)
    
    assert result == pytest.approx(expected)


def test_median_single_element():
    """Test median of single elements."""
    array = ak.Array([[5.0], [10.0]])
    
    result = ak.median(array, axis=1)
    expected = [5.0, 10.0]
    
    assert result.tolist() == pytest.approx(expected)


def test_median_even_odd_lengths():
    """Test median with both even and odd length sublists."""
    array = ak.Array([[1, 2, 3], [4, 5]])  # odd and even lengths
    
    result = ak.median(array, axis=1)
    expected = [2.0, 4.5]  # median of [1,2,3]=2, median of [4,5]=4.5
    
    assert result.tolist() == pytest.approx(expected)


def test_median_empty_array():
    """Test median of empty array."""
    array = ak.Array([])
    
    result = ak.median(array, axis=None)
    assert result is None


def test_median_all_empty_sublists():
    """Test median when all sublists are empty."""
    array = ak.Array([[], []])
    
    result = ak.median(array, axis=1, mask_identity=True)
    assert result.tolist() == [None, None]
    
    result = ak.median(array, axis=1, mask_identity=False)
    result_list = result.tolist()
    assert all(np.isnan(x) for x in result_list)


def test_median_numpy_compatibility():
    """Test that median works with numpy arrays via NEP-18."""
    np_array = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Test that np.median works on awkward arrays (via NEP-18)
    ak_array = ak.from_numpy(np_array)
    
    result_np = np.median(np_array, axis=1)
    result_ak = np.median(ak_array, axis=1)
    
    np.testing.assert_array_almost_equal(result_np, result_ak.to_numpy())