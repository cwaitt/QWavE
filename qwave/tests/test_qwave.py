"""
Unit and regression test for the qwave package.
"""

# Import package, test suite, and other packages as needed
import qwave
import pytest
import sys

def test_qwave_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qwave" in sys.modules
