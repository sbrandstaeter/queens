"""Test module for fcc utils."""

import pytest
from mock import Mock

from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.monte_carlo_iterator import MonteCarloIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.utils.exceptions import InvalidOptionError
from pqueens.utils.fcc_utils import (
    VALID_TYPES,
    check_for_reference,
    from_config_create_iterator,
    from_config_create_object,
    insert_new_obj,
)


@pytest.fixture
def config_1():
    """Dummy config 1."""
    config = {
        "a": "b",
        "c": {
            "d": "e",
            "plot_name": "pretty_plot",
            "f": {
                "g": "h",
            },
        },
    }
    return config


@pytest.fixture
def config_2():
    """Dummy config 2."""
    config = {
        "a": "b",
        "dummy_name": "dummy",
        "c": {"d": "e", "dummy_name": "another_dummy", "f": {"g": "h", "dummy_name": "dummy"}},
    }
    return config


@pytest.fixture
def parameters():
    """Dummy parameters."""
    return Mock()


@pytest.fixture
def dummy_obj():
    """Dummy object."""
    return Mock()


@pytest.fixture
def inserted_config_2(dummy_obj):
    """Dummy config_2 with inserted object."""
    config = {
        "a": "b",
        "dummy": dummy_obj,
        "c": {"d": "e", "dummy_name": "another_dummy", "f": {"g": "h", "dummy": dummy_obj}},
    }
    return config


def test_check_for_reference_false(config_1):
    """Test case for check_for_reference function."""
    assert not check_for_reference(config_1)


def test_check_for_reference_true_1(config_1):
    """Test case for check_for_reference function."""
    config_1['c']["dummy_name"] = "dummy"
    assert check_for_reference(config_1)


def test_check_for_reference_true_2(config_1):
    """Test case for check_for_reference function."""
    config_1["c"]["f"]["dummy_name"] = "dummy"
    assert check_for_reference(config_1)


def test_insert_new_obj(config_2, dummy_obj, inserted_config_2):
    """Test insert_new_obj function."""
    config = insert_new_obj(config_2, "dummy", dummy_obj)
    assert config == inserted_config_2


def test_from_config_create_object_iterator(mocker, config_1, parameters):
    """Test case for from_config_create_object function."""
    mp1 = mocker.patch("pqueens.utils.fcc_utils.get_module_class", return_value=MonteCarloIterator)
    mp2 = mocker.patch(
        "pqueens.iterators.monte_carlo_iterator.MonteCarloIterator.__init__", return_value=None
    )
    from_config_create_object(config_1, parameters)

    assert mp1.called_once_with(config_1, VALID_TYPES)
    assert mp2.call_args_list[0].kwargs == {**config_1, "parameters": parameters}
    assert not mp2.call_args_list[0].args
    assert mp2.call_count == 1


def test_from_config_create_object_model(mocker, config_1):
    """Test case for from_config_create_object function."""
    mp1 = mocker.patch("pqueens.utils.fcc_utils.get_module_class", return_value=SimulationModel)
    mp2 = mocker.patch(
        "pqueens.models.simulation_model.SimulationModel.__init__", return_value=None
    )
    from_config_create_object(config_1, parameters)

    assert mp1.called_once_with(config_1, VALID_TYPES)
    assert mp2.call_args_list[0].kwargs == config_1
    assert not mp2.call_args_list[0].args
    assert mp2.call_count == 1


def test_from_config_create_object_interface(mocker, config_1, parameters):
    """Test case for from_config_create_object function."""
    mp1 = mocker.patch(
        "pqueens.utils.fcc_utils.get_module_class", return_value=DirectPythonInterface
    )
    mp2 = mocker.patch(
        "pqueens.interfaces.direct_python_interface.DirectPythonInterface.__init__",
        return_value=None,
    )
    from_config_create_object(config_1, parameters)

    assert mp1.called_once_with(config_1, VALID_TYPES)
    assert mp2.call_args_list[0].kwargs == {**config_1, "parameters": parameters}
    assert not mp2.call_args_list[0].args
    assert mp2.call_count == 1


def test_from_config_create_iterator_runtime_error_case_1():
    """Test case for from_config_create_iterator function.

    Configuration fails due to missing 'method' description.
    """
    with pytest.raises(RuntimeError, match=r"Queens run can not be configured*"):
        from_config_create_iterator({'c': 'd'})


def test_from_config_create_iterator_runtime_error_case_2():
    """Test case for from_config_create_iterator function.

    Configuration fails due to circular dependencies.
    """
    config = {'a': {'b_name': 'd'}, 'd': {'e_name': 'a'}}
    with pytest.raises(RuntimeError, match=r"Queens run can not be configured*"):
        from_config_create_iterator(config)


def test_from_config_create_iterator_invalid_option_error_case_1():
    """Test case for from_config_create_iterator function.

    Configuration fails due to invalid class type 'bla'.
    """
    config = {'a': {'type': 'bla'}}
    with pytest.raises(InvalidOptionError, match="Object 'a' can not be initialized."):
        from_config_create_iterator(config)


def test_from_config_create_iterator_invalid_option_error_case_2():
    """Test case for from_config_create_iterator function.

    Configuration fails due to missing options for 'monte_carlo'.
    """
    config = {'a': {'type': 'monte_carlo'}}
    with pytest.raises(InvalidOptionError, match="Object 'a' can not be initialized."):
        from_config_create_iterator(config)
