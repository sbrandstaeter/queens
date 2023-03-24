"""Unit tests for gradient handler utils."""
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from pqueens.interfaces.job_interface import JobInterface
from pqueens.utils.gradient_handler import (
    AdjointGradient,
    FiniteDifferenceGradient,
    GradientHandler,
    ProvidedGradient,
    from_config_create_grad_handler,
    prepare_downstream_gradient_fun,
)
from pqueens.utils.valid_options_utils import InvalidOptionError


# ------------------------ some fixtures --------------------------#
@pytest.fixture()
def dummy_config():
    """A dummy config dict."""
    config = {
        "global_settings": {"experiment_name": "my_experiment"},
        "grad_handler": {
            "type": "finite_differences",
            "finite_difference_method": "2-point",
            "step_size": 1e-6,
            "gradient_interface_name": "gradient_interface",
            "upstream_gradient_file_name": "adjoint_grad_objective.csv",
        },
        "gradient_interface": {"type": "job_interface", "driver_name": "driver"},
        "resources": {
            "local_machine": {
                "scheduler_name": "local_scheduler",
                "max_concurrent": 1,
                "max_finished_jobs": 1000,
            }
        },
        "local_scheduler": {
            "experiment_dir": "experiment_dir",
            "type": "standard",
            "singularity": False,
            "num_procs": 1,
            "num_procs_post": 1,
        },
        "input_file": "dummy",
    }

    return config


@pytest.fixture()
def model_interface():
    """Fixture for dummy model interface."""
    interface_name = "interface_dummy"
    resources = "resources_dummy"
    experiment_name = "my_experiment"
    db = "my_db"
    polling_time = 1
    experiment_dir = "some_dir"
    remote = False
    remote_connect = None
    scheduler_type = None
    direct_scheduling = False
    time_for_data_copy = 1
    driver_name = "my_driver"
    experiment_field_name = "experiment_field_name"

    dummy_interface = JobInterface(
        interface_name,
        resources,
        experiment_name,
        db,
        polling_time,
        experiment_dir,
        remote,
        remote_connect,
        scheduler_type,
        direct_scheduling,
        time_for_data_copy,
        driver_name,
        experiment_field_name,
    )

    return dummy_interface


# -------------- tests parent class -------------------------------#
def test_from_config_create_grad_handler(dummy_config, model_interface):
    """Test the config method of the parent class."""
    dummy_interface = "dummy"
    # test wrong grad handler name
    with pytest.raises(ValueError):
        grad_handler_name = "wrong_name"
        grad_handler_obj = from_config_create_grad_handler(
            grad_handler_name, dummy_interface, dummy_config
        )

    # test config for finite differences
    grad_handler_name = "grad_handler"
    grad_handler_obj = from_config_create_grad_handler(
        grad_handler_name, dummy_interface, dummy_config
    )
    assert isinstance(grad_handler_obj, FiniteDifferenceGradient)

    # test config for callable grad
    dummy_config["grad_handler"]["type"] = "provided"
    grad_handler_obj = from_config_create_grad_handler(
        grad_handler_name, dummy_interface, dummy_config
    )
    assert isinstance(grad_handler_obj, ProvidedGradient)

    # test config for adjoint grad
    dummy_config["grad_handler"]["type"] = "adjoint"
    grad_handler_obj = from_config_create_grad_handler(
        grad_handler_name, model_interface, dummy_config
    )
    assert isinstance(grad_handler_obj, AdjointGradient)

    # test config for invalid type
    with pytest.raises(InvalidOptionError):
        dummy_config["grad_handler"]["type"] = "aaadjoint"
        grad_handler_obj = from_config_create_grad_handler(
            grad_handler_name, dummy_interface, dummy_config
        )


def test_calculate_downstream_gradient():
    """Test the calculate downstream gradient method."""
    upstream_gradient_fun = lambda samples, outputs: samples**2 + outputs
    samples = np.array([[1, 1], [2, 2]])
    response = np.array([[1], [2]])
    gradient_response_batch = np.array([[1, 2, 3], [4, 5, 6]])
    downstream_gradient_batch = GradientHandler.calculate_downstream_gradient_with_chain_rule(
        upstream_gradient_fun, samples, response, gradient_response_batch
    )
    expected_downstream_gradient_batch = np.array([[4, 8, 12], [48, 60, 72]])
    np.testing.assert_array_equal(downstream_gradient_batch, expected_downstream_gradient_batch)


# -------------- tests finite difference class ---------------------#
def test_init_fd():
    """Test init method of finite differences class."""
    method = "2-step"
    step_size = 1e-5
    model_interface = "dummy"
    bounds = [-np.inf, np.inf]
    my_grad_obj = FiniteDifferenceGradient(method, step_size, model_interface, bounds)
    assert isinstance(my_grad_obj, FiniteDifferenceGradient)
    assert my_grad_obj.method == method
    assert my_grad_obj.step_size == step_size
    assert my_grad_obj.model_interface == model_interface
    assert my_grad_obj.bounds == bounds


def test_fcc_fd(dummy_config):
    """Test fcc method from FD gradient class."""
    grad_name = "grad_handler"
    model_interface = "dummy"

    # check the valid options
    mygrad_obj = FiniteDifferenceGradient.from_config_create_grad_handler(
        grad_name, model_interface, dummy_config
    )
    assert isinstance(mygrad_obj, FiniteDifferenceGradient)
    assert mygrad_obj.method == dummy_config["grad_handler"]["finite_difference_method"]
    assert mygrad_obj.step_size == dummy_config["grad_handler"]["step_size"]
    assert mygrad_obj.model_interface == model_interface

    # try invalid finite difference method
    dummy_config["grad_handler"]["finite_difference_method"] = "some_invalid_method"
    with pytest.raises(InvalidOptionError):
        mygrad_obj = FiniteDifferenceGradient.from_config_create_grad_handler(
            grad_name, model_interface, dummy_config
        )


def test_evaluate_and_gradient_fd():
    """Test evaluate and gradient of finite difference method."""
    model_interface = "dummy"
    bounds = [-np.inf, np.inf]
    mygrad_obj = FiniteDifferenceGradient('2-point', 1e-5, model_interface, bounds)
    samples = np.array([1.0])

    def evaluate_fun(samples):
        """Dummy evaluate function."""
        out = []
        for sample in samples:
            out.append(sample**2)

        out = np.array(out).reshape(-1, 1)
        out_dict = {"mean": out}
        return out_dict

    def upstream_gradient_fun(_samples, outputs):
        """Dummy upstream gradient function."""
        out = []
        for y in outputs:
            out.append(y**2 + 1.0)
        out = np.array(out).reshape(-1, 1)
        return out

    # test wrong sample dimensions
    with pytest.raises(ValueError):
        response, gradient_response = mygrad_obj.evaluate_and_gradient(
            samples=samples, evaluate_fun=evaluate_fun
        )

    # test without upstream gradient function
    samples = np.array([[1.0]])
    response, gradient_response = mygrad_obj.evaluate_and_gradient(
        samples=samples, evaluate_fun=evaluate_fun
    )
    assert response == np.array([[1.0]])
    np.testing.assert_array_almost_equal(gradient_response, np.array([[2.0]]), decimal=1e-5)

    # test with upstream gradient function
    response, gradient_response = mygrad_obj.evaluate_and_gradient(
        samples=samples, evaluate_fun=evaluate_fun, upstream_gradient_fun=upstream_gradient_fun
    )
    assert response == 1.0
    np.testing.assert_array_almost_equal(gradient_response, np.array([4.0]), decimal=1e-5)

    # test for 2D function without upstream gradient
    samples = np.array([[1.0, 1.0]])

    def evaluate_fun_2d(samples):
        """A dummy eval fun.

        in 2d.
        """
        result_lst = []
        for x in samples:
            result_lst.append(x[0] ** 2 + x[1] ** 2)
        result = np.array(result_lst)
        result_dict = {"mean": result}
        return result_dict

    response, gradient_response = mygrad_obj.evaluate_and_gradient(
        samples=samples, evaluate_fun=evaluate_fun_2d
    )
    assert response == 2.0
    np.testing.assert_array_almost_equal(gradient_response, np.array([2.0, 2.0]), decimal=1e-5)


# -------------- tests provided gradient class ---------------------#
def test_init_provided():
    """Test init method of callable class."""
    model_interface = None
    _get_model_output_fun = ProvidedGradient._get_output_without_gradient_interface
    my_grad_obj = ProvidedGradient(model_interface, _get_model_output_fun)
    assert isinstance(my_grad_obj, ProvidedGradient)
    assert my_grad_obj._get_model_output == ProvidedGradient._get_output_without_gradient_interface


def test_fcc_provided(dummy_config):
    """Test fcc method from callable gradient class."""
    grad_name = "grad_handler"
    model_interface = "dummy"
    dummy_config[grad_name]["type"] = "provided"

    # check the valid options with gradient interface
    mygrad_obj = ProvidedGradient.from_config_create_grad_handler(
        grad_name, model_interface, dummy_config
    )
    assert isinstance(mygrad_obj, ProvidedGradient)
    assert mygrad_obj._get_model_output != ProvidedGradient._get_output_without_gradient_interface

    # dont provide a gradient interface
    dummy_config[grad_name]["gradient_interface_name"] = None
    mygrad_obj = ProvidedGradient.from_config_create_grad_handler(
        grad_name, model_interface, dummy_config
    )
    assert mygrad_obj.model_interface == model_interface
    assert mygrad_obj._get_model_output == ProvidedGradient._get_output_without_gradient_interface


def test_evaluate_and_gradient_provided(dummy_config, mocker):
    """Test evaluate and gradient method for callable gradient."""
    grad_name = "grad_handler"
    model_interface = "dummy"

    dummy_config[grad_name]["type"] = "callable"
    mygrad_obj = ProvidedGradient.from_config_create_grad_handler(
        grad_name, model_interface, dummy_config
    )

    samples = np.array([[1.0]])

    def eval_fun(x):
        """Dummy eval fun."""
        f = x**2
        df = 2 * x
        return {"mean": f, "gradient": df}

    mocker.patch(
        "pqueens.interfaces.job_interface.JobInterface.evaluate", return_value={"mean": 99}
    )

    # check evaluation with a gradient interface
    response, gradient_response = mygrad_obj.evaluate_and_gradient(
        samples=samples, evaluate_fun=eval_fun
    )
    assert response == 1.0
    assert gradient_response == 99

    # check evaluation without a gradient interface
    dummy_config["grad_handler"]["gradient_interface_name"] = None
    mygrad_obj = ProvidedGradient.from_config_create_grad_handler(
        grad_name, model_interface, dummy_config
    )
    response, gradient_response = mygrad_obj.evaluate_and_gradient(
        samples=samples, evaluate_fun=eval_fun
    )
    assert response == 1.0
    assert gradient_response == 2.0

    # check evaluation with grad objective function
    upstream_gradient_fun = lambda x, y: 2 * y
    response, gradient_response = mygrad_obj.evaluate_and_gradient(
        samples=samples, evaluate_fun=eval_fun, upstream_gradient_fun=upstream_gradient_fun
    )
    assert response == 1.0
    assert gradient_response == 4.0


def test_get_output_without_gradient_interface():
    """Test get output without gradient interface."""
    samples = np.array([[1.0], [2.0]])
    evaluate_fun = lambda x: {"mean": x**2, "gradient": 2 * x}
    response, gradient_response_batch = ProvidedGradient._get_output_without_gradient_interface(
        samples, evaluate_fun
    )
    np.testing.assert_array_equal(response, np.array([[1.0], [4.0]]))
    np.testing.assert_array_equal(gradient_response_batch, np.array([[2.0], [4.0]]))


def test_get_output_with_gradient_interface(mocker):
    """Test get output with gradient interface."""
    samples = np.array([[1.0], [2.0]])
    evaluate_fun = lambda x: {"mean": x**2, "gradient": 2 * x}
    gradient_interface = mocker.MagicMock()
    gradient_interface.evaluate.return_value = {"mean": np.array([[2.0], [4.0]])}

    response, gradient_response_batch = ProvidedGradient._get_output_with_gradient_interface(
        gradient_interface, samples, evaluate_fun
    )

    np.testing.assert_array_equal(response, np.array([[1.0], [4.0]]))
    np.testing.assert_array_equal(gradient_response_batch, np.array([[2.0], [4.0]]))


# -------------- tests adjoint class --------------------------------#
def test_init_adjoint(model_interface):
    """Test init method of adjoint class."""
    adjoint_file_name = "file"
    gradient_interface = "my_gradient_interface"
    experiment_name = "experiment_name_dummy"
    my_grad_obj = AdjointGradient(
        adjoint_file_name, gradient_interface, experiment_name, model_interface
    )
    assert isinstance(my_grad_obj, AdjointGradient)
    assert my_grad_obj.upstream_gradient_file_name == adjoint_file_name
    assert my_grad_obj.gradient_interface == gradient_interface
    assert my_grad_obj.experiment_name == experiment_name
    assert my_grad_obj.model_interface == model_interface


def test_fcc_adjoint(dummy_config, model_interface):
    """Test fcc method from adjoint gradient class."""
    grad_name = "grad_handler"
    dummy_config[grad_name]["type"] = "adjoint"

    # check the valid options with gradient interface and no adjoint file name
    mygrad_obj = AdjointGradient.from_config_create_grad_handler(
        grad_name, model_interface, dummy_config
    )
    assert isinstance(mygrad_obj, AdjointGradient)
    assert isinstance(mygrad_obj.gradient_interface, JobInterface)
    assert mygrad_obj.upstream_gradient_file_name == "adjoint_grad_objective.csv"
    assert mygrad_obj.model_interface == model_interface

    # provide a specific adjoint file name
    dummy_config[grad_name]["upstream_gradient_file_name"] = "another_file_name.csv"
    mygrad_obj = AdjointGradient.from_config_create_grad_handler(
        grad_name, model_interface, dummy_config
    )
    assert isinstance(mygrad_obj, AdjointGradient)
    assert isinstance(mygrad_obj.gradient_interface, JobInterface)
    assert mygrad_obj.upstream_gradient_file_name == "another_file_name.csv"

    # check if error is raised if interface is not a JobInterface
    dummy_interface = "dummy_interface"
    with pytest.raises(NotImplementedError):
        mygrad_obj = AdjointGradient.from_config_create_grad_handler(
            grad_name, dummy_interface, dummy_config
        )

    # remove the gradient interface and check if error is raised
    dummy_config[grad_name]["gradient_interface_name"] = None
    with pytest.raises(ValueError):
        mygrad_obj = AdjointGradient.from_config_create_grad_handler(
            grad_name, model_interface, dummy_config
        )


def test_evaluate_and_gradient_adjoint(dummy_config, mocker):
    """Test evaluate and gradient for adjoint grad object."""
    # create some named tuples to mimic the grad interface
    grad_interface_dummy = namedtuple(
        "interface",
        ["experiment_dir", "evaluate", "batch_number", "experiment_field_name", "job_ids"],
    )
    grad_interface = grad_interface_dummy(
        experiment_dir=Path("some_experiment_dir"),
        evaluate=lambda x: {"mean": 2 * x},
        batch_number=1,
        experiment_field_name="test",
        job_ids=[1, 2],
    )

    # specify further class arguments
    upstream_gradient_file_name = "adjoint_grad_objective.csv"
    gradient_interface = grad_interface
    experiment_name = "test"
    model_interface = grad_interface
    grad_obj = AdjointGradient(
        upstream_gradient_file_name, gradient_interface, experiment_name, model_interface
    )

    samples = np.array([[1.0]])
    evaluate_fun = lambda x: {"mean": x**2}

    # check if error is raised when no upstram grad fun is provided
    with pytest.raises(RuntimeError):
        response, gradient_response = grad_obj.evaluate_and_gradient(
            samples=samples, evaluate_fun=evaluate_fun
        )

    # now check a valid evaluation and mock the respective methods
    upstream_gradient_fun = lambda x, y: 2 * y
    m1 = mocker.patch("pqueens.utils.gradient_handler.write_to_csv")
    response, gradient_response = grad_obj.evaluate_and_gradient(
        samples=samples, evaluate_fun=evaluate_fun, upstream_gradient_fun=upstream_gradient_fun
    )

    assert response == 1.0
    assert gradient_response == 2.0
    m1.assert_called_once()


def test_prepare_downstream_gradient_fun():
    """Test the composition of the downstream function's gradient."""
    sub_model_eval = lambda x: x + 1  # returns y, dependent on x
    eval_output_fun = lambda y: y**2  # returns l, dependent on x and y
    partial_grad_evaluate_fun = lambda x, y: 2 * y  # returns dl/dy, dependent on x, y
    upstream_gradient_fun = lambda x, l: np.cos(l)  # upstream gradient, dependent on x, l

    # generate composed function
    composed_grad = prepare_downstream_gradient_fun(
        eval_output_fun=eval_output_fun,
        partial_grad_evaluate_fun=partial_grad_evaluate_fun,
        upstream_gradient_fun=upstream_gradient_fun,
    )

    # test the composed function
    samples = np.array([[1.0]])
    sub_model_output = sub_model_eval(samples)

    grad_value = composed_grad(samples, sub_model_output)

    # upstream gradient fun: do/dl  --> we need to compose do/dy and hand this down
    # do/dy = do/dl * dl/dy --> here: cos(l) * 2y --> make everything only dependent on x, y
    # --> cos(y**2) * 2y, with x=1 and y=x+1 this should evaluate to cos(2**2) * 4
    assert grad_value == np.cos(4) * 4
