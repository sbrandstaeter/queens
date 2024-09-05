# -*- coding: utf-8 -*-
"""Interfaces.

This package contains a set of so-called interfaces. The purpose of an
interface is essentially the mapping of inputs to outputs. For now there
are four kinds of interfaces plus the base class.

The mapping is made by passing the inputs further down to a
*regression_approximation* or a *mf_regression_approximation*, both of
which essentially then evaluate a regression model themselves.

The alternatives are the evaluation of simple python function using the
*direct_python_interface* or running an external software through the
*job_interface*.
"""

from queens.interfaces.bmfia_interface import BmfiaInterface
from queens.interfaces.job_interface import JobInterface

VALID_TYPES = {
    "job_interface": JobInterface,
    "bmfia_interface": BmfiaInterface,
}
