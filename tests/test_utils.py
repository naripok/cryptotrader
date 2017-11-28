"""
Test utility functions
"""
import os
import shutil
import pytest
import mock
from hypothesis import given, example, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
from math import nan
import numpy as np

from cryptotrader.utils import convert_to, array_normalize, array_softmax
from decimal import Decimal, InvalidOperation, Overflow

@given(st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.integers()))
def test_convert_to(data):
    if abs(data) < Decimal('1e18'):
        number = convert_to.decimal(data)
        assert number - Decimal.from_float(data).quantize(Decimal('0e-9')) < Decimal("1e-8")

    elif abs(data) < Decimal('1e33'):
        with pytest.raises(InvalidOperation):
            convert_to.decimal(data)
    else:
        with pytest.raises(Overflow):
            convert_to.decimal(data)

@given(arrays(dtype=np.float32,
              shape=array_shapes(),
              elements=st.floats(allow_nan=False, allow_infinity=False, max_value=1e6, min_value=-1e6)))
def test_array_normalize(data):
    array_normalize(data)

@given(arrays(dtype=np.float32,
              shape=array_shapes(),
              elements=st.floats(allow_nan=False, allow_infinity=False, max_value=1e14, min_value=-1e14)))
def test_array_softmax(data):
    array_softmax(data)


if __name__ == '__main__':
    pytest.main()