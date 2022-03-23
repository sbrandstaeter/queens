# -*- coding: utf-8 -*-
"""Interfaces.

This package contains a set of so-called interfaces. The purpose of an interface
is essentially the mapping of inputs to outputs. For now there are four kinds
of interface plus the base class.

The mapping is can made by passing the inputs furhter down to a
regression_approximation or a mf_regression_approximation, both of which
essentially then evaluate a regression modelself.

The alternatives are the evaluation of simple python function using the
direct_python_interface or running an external software through the job_interface.
"""


def from_config_create_interface(interface_name, config):
    """Create Interface from config dictionary.

    Args:
        interface_name (str):   name of the interface
        config (dict):          dictionary with problem description

    Returns:
        interface:              Instance of one of the derived interface classes
    """
    from pqueens.interfaces.approximation_interface import ApproximationInterface
    from pqueens.interfaces.approximation_interface_mf import ApproximationInterfaceMF
    from pqueens.interfaces.bmfmc_interface import BmfmcInterface
    from pqueens.interfaces.direct_python_interface import DirectPythonInterface
    from pqueens.interfaces.job_interface import JobInterface

    interface_dict = {
        'job_interface': JobInterface,
        'direct_python_interface': DirectPythonInterface,
        'approximation_interface': ApproximationInterface,
        'approximation_interface_mf': ApproximationInterfaceMF,
        'bmfmc_interface': BmfmcInterface,
    }

    interface_options = config[interface_name]
    # determine which object to create
    interface_class = interface_dict[interface_options["type"]]

    # get the driver which belongs to the model/interface
    # (important if several models are involved)
    driver_name = interface_options.get("driver")
    return interface_class.from_config_create_interface(
        interface_name, config, driver_name=driver_name
    )
