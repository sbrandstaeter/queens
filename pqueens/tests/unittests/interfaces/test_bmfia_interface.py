"""Unittests for the Bayesian multi-fidelity inverse analysis interface."""

import numpy as np
import pytest

from pqueens.interfaces.bmfia_interface import BmfiaInterface


# ---- Fixtures and helper methods / classes ---------
@pytest.fixture
def default_bmfia_interface():
    """Fixture for a dummy bmfia interface."""
    pass


# ---- Actual unittests ------------------------------
@pytest.mark.unit_tests
def test__init__(default_bmfia_interface):
    """Test the instantiation of the interface object."""
    pass


@pytest.mark.unit_tests
def test_map(default_bmfia_interface):
    """Test the mapping for the multi-fidelity interface."""
    pass


@pytest.mark.unit_tests
def test_build_approximation(default_bmfia_interface):
    """Test the set-up / build of the probabilsitic regression models."""
    pass
