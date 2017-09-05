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
from decimal import Decimal, InvalidOperation

@given(st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.integers()))
def test_convert_to(data):
    if abs(data) < Decimal('1e17'):
        number = convert_to.decimal(data)
        assert number == Decimal.from_float(data).quantize(Decimal('1e-12'))
    else:
        with pytest.raises(InvalidOperation):
            convert_to.decimal(data)


@given(arrays(dtype=np.float32,
              shape=array_shapes(),
              elements=st.floats(allow_nan=False, allow_infinity=False, max_value=1e15, min_value=-1e15)))
def test_array_normalize(data):
    array_normalize(data)

@given(arrays(dtype=np.float32,
              shape=array_shapes(),
              elements=st.floats(allow_nan=False, allow_infinity=False, max_value=1e15, min_value=-1e15)))
def test_array_softmax(data):
    array_softmax(data)


if __name__ == '__main__':
    pytest.main()