"""
Test utility functions
"""
import os
import shutil
import pytest
import mock
from hypothesis import given, example, strategies as st
from math import nan

from cryptotrader.utils import convert_to
from decimal import Decimal, InvalidOperation

@given(st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.integers()))
def test_convert_to(data):
    if abs(data) < Decimal('1e17'):
        number = convert_to.decimal(data)
        assert number == Decimal.from_float(data).quantize(Decimal('1e-12'))
    else:
        with pytest.raises(InvalidOperation):
            convert_to.decimal(data)


if __name__ == '__main__':
    pytest.main()