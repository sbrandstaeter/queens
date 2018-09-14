# -*- coding: utf-8 -*-
"""
------------
Interfaces
------------

This package contains a set of so-called interfaces. The purpose of an interface
is essentially the mapping of inputs to outputs. For now there are four kinds
of interface plus the base class.

The mapping is can made by passing the inputs furhter down to a
regression_approximation or a mf_regression_approximation, both of which
essentially then evaluate a regression modelself.

The alternatives are the evaluation of simple python function using the
direct_python_interface or running an external software through the job_interface.

"""
