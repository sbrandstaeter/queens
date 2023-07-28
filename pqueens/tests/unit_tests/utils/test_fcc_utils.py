"""Test module for fcc utils."""
from copy import deepcopy

import pytest
from mock import Mock

from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.monte_carlo_iterator import MonteCarloIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.schedulers.local_scheduler import LocalScheduler
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
def global_settings():
    """Dummy global_settings."""
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
    assert check_for_reference(config_1) is False


def test_check_for_reference_true_1(config_1):
    """Test case for check_for_reference function."""
    config = deepcopy(config_1)
    config["dummy_name"] = "dummy"
    assert check_for_reference(config) is True


def test_check_for_reference_true_2(config_1):
    """Test case for check_for_reference function."""
    config = deepcopy(config_1)
    config["c"]["f"]["dummy_name"] = "dummy"
    assert check_for_reference(config) is True


def test_insert_new_obj(config_2, dummy_obj, inserted_config_2):
    """Test insert_new_obj function."""
    config = deepcopy(config_2)
    config = insert_new_obj(config, "dummy", dummy_obj)
    assert config == inserted_config_2


def test_from_config_create_object_1(mocker, config_1, global_settings, parameters):
    """Test case for from_config_create_object function."""
    config = deepcopy(config_1)
    config_ = deepcopy(config_1)
    mp1 = mocker.patch("pqueens.utils.fcc_utils.get_module_class", return_value=MonteCarloIterator)
    mp2 = mocker.patch(
        "pqueens.iterators.monte_carlo_iterator.MonteCarloIterator.__init__", return_value=None
    )
    from_config_create_object(config, global_settings, parameters)

    assert mp1.called_once_with(config_, VALID_TYPES)
    assert mp2.call_args_list[0].kwargs == {
        **config_,
        "parameters": parameters,
        "global_settings": global_settings,
    }
    assert not mp2.call_args_list[0].args
    assert mp2.call_count == 1


def test_from_config_create_object_2(mocker, config_1):
    """Test case for from_config_create_object function."""
    config = deepcopy(config_1)
    mp1 = mocker.patch("pqueens.utils.fcc_utils.get_module_class", return_value=SimulationModel)
    mp2 = mocker.patch(
        "pqueens.models.simulation_model.SimulationModel.__init__", return_value=None
    )
    from_config_create_object(config, global_settings, parameters)

    assert mp1.called_once_with(config, VALID_TYPES)
    assert mp2.call_args_list[0].kwargs == config
    assert not mp2.call_args_list[0].args
    assert mp2.call_count == 1


def test_from_config_create_object_3(mocker, config_1, parameters):
    """Test case for from_config_create_object function."""
    config = deepcopy(config_1)
    config_ = deepcopy(config_1)
    mp1 = mocker.patch(
        "pqueens.utils.fcc_utils.get_module_class", return_value=DirectPythonInterface
    )
    mp2 = mocker.patch(
        "pqueens.interfaces.direct_python_interface.DirectPythonInterface.__init__",
        return_value=None,
    )
    from_config_create_object(config, global_settings, parameters)

    assert mp1.called_once_with(config_, VALID_TYPES)
    assert mp2.call_args_list[0].kwargs == {**config_, "parameters": parameters}
    assert not mp2.call_args_list[0].args
    assert mp2.call_count == 1


def test_from_config_create_object_4(mocker, config_1, parameters):
    """Test case for from_config_create_object function."""
    config = deepcopy(config_1)
    config_ = deepcopy(config_1)
    mp1 = mocker.patch("pqueens.utils.fcc_utils.get_module_class", return_value=LocalScheduler)
    mp2 = mocker.patch(
        "pqueens.schedulers.local_scheduler.LocalScheduler.__init__", return_value=None
    )
    from_config_create_object(config, global_settings, parameters)

    assert mp1.called_once_with(config_, VALID_TYPES)
    assert mp2.call_args_list[0].kwargs == {**config_, "global_settings": global_settings}
    assert not mp2.call_args_list[0].args
    assert mp2.call_count == 1


def test_from_config_create_iterator_1():
    """Test case for from_config_create_iterator function."""
    with pytest.raises(RuntimeError, match=r'Queens run can not be configured!'):
        from_config_create_iterator({'global_settings': 'b', 'c': 'd'})


def test_from_config_create_iterator_2():
    """Test case for from_config_create_iterator function."""
    config = {'global_settings': {}, 'a': {'b_name': 'c'}, 'd': {'e_name': 'a'}}
    with pytest.raises(RuntimeError, match=r'Queens run can not be configured!'):
        from_config_create_iterator(config)


def test_from_config_create_iterator_3():
    """Test case for from_config_create_iterator function."""
    config = {
        'global_settings': {},
        'a': {'type': 'bla'},
    }
    with pytest.raises(InvalidOptionError, match="Object 'a' can not be initialized."):
        from_config_create_iterator(config)


def test_from_config_create_iterator_4():
    """Test case for from_config_create_iterator function."""
    config = {
        'global_settings': {},
        'a': {'type': 'monte_carlo'},
    }
    with pytest.raises(InvalidOptionError, match="Object 'a' can not be initialized."):
        from_config_create_iterator(config)
