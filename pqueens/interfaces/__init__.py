# -*- coding: utf-8 -*-
"""Interfaces.

This package contains a set of so-called interfaces. The purpose of an interface
is essentially the mapping of inputs to outputs. For now there are four kinds
of interface plus the base class.

The mapping is can made by passing the inputs further down to a
regression_approximation or a mf_regression_approximation, both of which
essentially then evaluate a regression model themselves.

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
    from pqueens.interfaces.direct_python_interface import DirectPythonInterface
    from pqueens.interfaces.job_interface import JobInterface
    from pqueens.utils.import_utils import get_module_attribute
    from pqueens.utils.valid_options_utils import get_option

    interface_dict = {
        'job_interface': JobInterface,
        'direct_python_interface': DirectPythonInterface,
        'approximation_interface': ApproximationInterface,
        'approximation_interface_mf': ApproximationInterfaceMF,
    }

    interface_options = config[interface_name]
    # determine which object to create
    if interface_options.get("external_python_module"):
        module_path = interface_options["external_python_module"]
        module_attribute = interface_options.get("type")
        interface_class = get_module_attribute(module_path, module_attribute)
    else:
        interface_class = get_option(interface_dict, interface_options.get("type"))

    return interface_class.from_config_create_interface(interface_name, config)
