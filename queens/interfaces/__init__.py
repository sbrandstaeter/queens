# -*- coding: utf-8 -*-
"""Interfaces.

This package contains a set of so-called interfaces. The purpose of an
interface is essentially the mapping of inputs to outputs.

The mapping is made by passing the inputs further down to a
*regression_approximation* or a *mf_regression_approximation*, both of
which essentially then evaluate a regression model themselves.
"""

from queens.interfaces.bmfia_interface import BmfiaInterface

VALID_TYPES = {
    "bmfia_interface": BmfiaInterface,
}
