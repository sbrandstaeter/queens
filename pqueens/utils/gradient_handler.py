"""Module that collects functionalities for model gradient calculation."""

import abc
import logging

import numpy as np

from pqueens.interfaces import from_config_create_interface
from pqueens.interfaces.dask_job_interface import JobInterface as DaskJobInterface
from pqueens.interfaces.job_interface import JobInterface
from pqueens.utils.config_directories import current_job_directory
from pqueens.utils.fd_jacobian import fd_jacobian, get_positions
from pqueens.utils.import_utils import get_module_class
from pqueens.utils.io_utils import write_to_csv
from pqueens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)

VALID_TYPES = {
    "finite_differences": ["pqueens.utils.gradient_handler", "FiniteDifferenceGradient"],
    "provided": ["pqueens.utils.gradient_handler", "ProvidedGradient"],
    "adjoint": ["pqueens.utils.gradient_handler", "AdjointGradient"],
}


def from_config_create_grad_handler(grad_obj_name, model_interface, config):
    """Create gradient handler object form config dictionary.

    Args:
        grad_obj_name (str): Name of the gradient object.
        model_interface (obj): Interface of the associated model.
        config (dict): Dictionary with problem description.
    """
    grad_options = config.get(grad_obj_name)
    if grad_options is None:
        raise ValueError(
            "The name of the gradient handler could not be found in the input file!"
            f"You provided the following name: {grad_obj_name}."
        )
    grad_class = get_module_class(grad_options, VALID_TYPES)
    grad_obj = grad_class.from_config_create_grad_handler(grad_obj_name, model_interface, config)
    return grad_obj


class GradientHandler(metaclass=abc.ABCMeta):
    """Abstract gradient handler class.

    Attributes:
        model_interface (Interface): Interface of the associated model.
    """

    def __init__(self, model_interface):
        """Initialize the gradient handler.

        Args:
            model_interface (Interface): Interface of the associated model.
        """
        self.model_interface = model_interface

    @abc.abstractmethod
    def evaluate_and_gradient(self, samples, evaluate_fun, upstream_gradient_fun=None):
        """Evaluate the model and its gradient at the given samples.

        Args:
            samples (np.array): Current samples at which model should be evaluated.
            evaluate_fun (obj): Evaluation function that runs the underlying model for the samples
            upstream_gradient_fun (optional, obj): An optional gradient function of an objective
                                                   function which is dependent on the samples and
                                                   the underlying model response (which can be
                                                   evaluated using the evaluate_fun)
        """
        pass

    @staticmethod
    def calculate_downstream_gradient_with_chain_rule(
        upstream_gradient_fun, samples, response, gradient_response_batch
    ):
        """Calculate the downstream gradient using the chain rule.

        Args:
            upstream_gradient_fun (function): Function that calculates the gradient of the upstream
                                              objective function
            samples (np.array): Input samples
            response (np.array): Model response at the given input samples

        Returns:
            downstream_gradient_batch (np.array): Downstream gradient at the given input samples
        """
        gradient_upstream_d_y_batch = upstream_gradient_fun(samples, response)

        # check if the gradient is a scalar and reshape it to a vector
        if gradient_response_batch[0].ndim < 2:
            gradient_response_batch = [
                gradient_response.reshape(-1, 1) for gradient_response in gradient_response_batch
            ]

        downstream_gradient_batch = np.einsum(
            "ij,ikj->ik", gradient_upstream_d_y_batch, gradient_response_batch
        ).squeeze()

        return downstream_gradient_batch


class FiniteDifferenceGradient(GradientHandler):
    """Compute gradients based on finite differences.

    Attributes:
        method (str): Method to calculate a finite difference
                       based approximation of the Jacobian matrix:

                        - '2-point': a one sided scheme by definition
                        - '3-point': more exact but needs twice as many function evaluations
        step_size (float): Step size for the finite difference
                           approximation
    """

    def __init__(self, method, step_size, model_interface, bounds):
        """Initialize the finite difference gradient object.

        Args:
            method (str): Method to calculate a finite difference
                          based approximation of the Jacobian matrix:

                            - '2-point': a one sided scheme by definition
                            - '3-point': more exact but needs twice as many function evaluations
            step_size (float): Step size for the finite difference
                                approximation
            model_interface (obj): Interface of the associated model.
            bounds (tuple of array_like, optional): Lower and upper bounds on independent variables.
                                                    Defaults to no bounds meaning: [-inf, inf]
                                                    Each bound must match the size of *x0* or be a
                                                    scalar, in the latter case the bound will be the
                                                    same for all variables. Use it to limit the
                                                    range of function evaluation.
        """
        super().__init__(model_interface)
        self.method = method
        self.step_size = step_size
        self.bounds = bounds

    @classmethod
    def from_config_create_grad_handler(cls, grad_obj_name, model_interface, config):
        """Create the gradient object form the problem description.

        Args:
            grad_obj_name (str): Name of the gradient object.
            model_interface (obj): Interface of the associated model.
            config (dict): Dictionary with problem description.
        """
        VALID_FINITE_DIFFERENCE_METHODS = {
            "2-point": "2-point",
            "3-point": "3-point",
            "cs": "cs",
        }
        grad_options = config[grad_obj_name]
        finite_difference_type = grad_options.get("finite_difference_method")
        method = get_option(VALID_FINITE_DIFFERENCE_METHODS, finite_difference_type)
        step_size = grad_options.get("step_size", 1e-5)
        bounds = grad_options.get("bounds", [-np.inf, np.inf])
        _logger.debug(
            "The gradient calculation via finite differences uses the default step size of %s.",
            step_size,
        )
        return cls(method, step_size, model_interface, bounds)

    def evaluate_and_gradient(self, samples, evaluate_fun, upstream_gradient_fun=None):
        """Evaluate model and its gradient at given samples based on FD.

        Args:
            samples (np.array): Current samples at which model should be evaluated.
            evaluate_fun (obj): Evaluation function that runs the underlying model for the samples
            upstream_gradient_fun (optional, obj): An optional gradient function of an objective
                                                   function which is dependent on the samples and
                                                   the underlying model response (which can be
                                                   evaluated using the evaluate_fun)

        Returns:
            response (np.array): Array with model response for given input samples
            gradient_response (np.array): Array with row-wise model/objective fun gradients for
                                       given samples.
        """
        # check dimensions of samples
        if samples.ndim < 2:
            raise ValueError(
                "The sample dimension must be at least 2D! Columns represent different "
                "variable dimensions and rows different sample realizations."
            )

        num_samples = samples.shape[0]

        # calculate the additional sample points for the stencil per sample
        stencil_samples, delta_positions = zip(
            *[
                get_positions(
                    sample,
                    method=self.method,
                    rel_step=self.step_size,
                    bounds=self.bounds,
                )
                for sample in samples
            ]
        )
        stencil_samples = np.concatenate(stencil_samples)
        delta_positions = np.concatenate(delta_positions)

        # stack samples and stencil points and evaluate entire batch
        combined_samples = np.vstack((samples, stencil_samples))
        all_responses = evaluate_fun(combined_samples)["mean"]

        # make sure the dim of the array is at least 2d (note: np.atleast_2d would not work here,
        # as it transposes the array if ndim > 1 which is not desired.)
        if all_responses.ndim < 2:
            all_responses = all_responses.reshape(-1, 1)

        response = all_responses[:num_samples, :]
        additional_response_lst = np.array_split(all_responses[num_samples:, :], num_samples)

        # calculate the model gradients re-using the already computed model responses
        model_gradients_lst = []
        for output, delta_positions, additional_model_output_stencil in zip(
            response, delta_positions, additional_response_lst
        ):
            model_gradients_lst.append(
                fd_jacobian(
                    output.reshape(1, -1),
                    additional_model_output_stencil,
                    delta_positions,
                    False,
                    method=self.method,
                )
            )

        gradient_response_batch = np.atleast_3d(np.array(model_gradients_lst))

        # use chain rule to calculate the gradient in presents of an upstream fun
        if upstream_gradient_fun:
            gradient_response_batch = np.transpose(gradient_response_batch, (0, 2, 1))
            gradient_response_batch = GradientHandler.calculate_downstream_gradient_with_chain_rule(
                upstream_gradient_fun, samples, response, gradient_response_batch
            )

        return response, gradient_response_batch.squeeze()


class AdjointGradient(GradientHandler):
    """Gradients based on adjoint formulation.

    Gradients are calculated w.r.t. the model output in QUEENS, meaning the output of
    the simulation that was read in and filtered by the data-processor.

    Attributes:
        upstream_gradient_file_name (str): Name of the adjoint file that contains the evaluated
                                           derivative of the upstream functional/objective
                                           w.r.t. to the simulation output.
        gradient_interface (obj): Interface object for the adjoint simulation run.
    """

    def __init__(
        self, upstream_gradient_file_name, gradient_interface, experiment_name, model_interface
    ):
        """Initialize a AdjointGrad object.

        Args:
            upstream_gradient_file_name (str): Name of the adjoint file that contains the evaluated
                                               derivative of the functional w.r.t. to the
                                               simulation output.
            gradient_interface (obj): Interface object for the adjoint simulation run.
            experiment_name (str): Name of the current QUEENS experiment
            model_interface (obj): Interface of the associated model.
        """
        super().__init__(model_interface)
        self.gradient_interface = gradient_interface
        self.upstream_gradient_file_name = upstream_gradient_file_name
        self.experiment_name = experiment_name

    @classmethod
    def from_config_create_grad_handler(cls, grad_obj_name, model_interface, config):
        """Create the gradient object form the problem description.

        Args:
            grad_obj_name (str): Name of the gradient object.
            model_interface (obj): Interface of the associated model.
            config (dict): Dictionary with problem description.
        """
        # check if model interface is of type `job_interface`
        if not isinstance(model_interface, (JobInterface, DaskJobInterface)):
            raise NotImplementedError(
                "The adjoint-based gradient is at the moment only available for executables"
                "which use the 'JobInterface'."
            )

        grad_options = config[grad_obj_name]
        upstream_gradient_file_name = grad_options.get("upstream_gradient_file_name")
        if upstream_gradient_file_name is None:
            raise ValueError(
                "You must provide the key-value pair for `upstream_gradient_file_name` in"
                " the gradient handler object"
            )

        gradient_interface_name = grad_options.get("gradient_interface_name")
        if gradient_interface_name is None:
            raise ValueError(
                "You must provide the key-value pair for `gradient_interface_name` in"
                " the gradient handler object!"
            )
        gradient_interface = from_config_create_interface(gradient_interface_name, config)
        experiment_name = config["global_settings"]["experiment_name"]

        return cls(
            upstream_gradient_file_name, gradient_interface, experiment_name, model_interface
        )

    def evaluate_and_gradient(self, samples, evaluate_fun, upstream_gradient_fun=None):
        """Evaluate model gradient based on adjoints.

            samples (np.array): Current samples at which model should be evaluated.
            evaluate_fun (obj): Evaluation function that runs the underlying model for the samples
            upstream_gradient_fun (optional, obj): An optional gradient function of an objective
                                                   function which is dependent on the samples and
                                                   the underlying model response (which can be
                                                   evaluated using the evaluate_fun)

        Returns:
            response (np.array): Array with model response for given input samples
            gradient_response (np.array): Array with row-wise model/objective fun gradients for
                                       given samples.
        """
        # check if the upstream gradient of the objective w.r.t. to the model output was provided
        if upstream_gradient_fun is None:
            raise RuntimeError(
                "You must provide a upstream gradient function of the objective w.r.t. the model"
                "output!"
            )

        # evaluate the forward model
        response = evaluate_fun(samples)["mean"]

        # calculate the upstream gradient for entire batch
        upstream_gradient_batch = upstream_gradient_fun(samples, response)

        num_samples = response.shape[0]
        # get last job_ids
        if isinstance(self.model_interface, DaskJobInterface):
            last_job_ids = [
                self.model_interface.latest_job_id - num_samples + i + 1 for i in range(num_samples)
            ]
            experiment_dir = self.gradient_interface.scheduler.experiment_dir
        else:
            last_job_ids = self.model_interface.job_ids[-response.shape[0] :]
            experiment_dir = self.gradient_interface.experiment_dir

        # write adjoint data for each sample to adjoint files in old job directories
        for job_id, grad_y_objective in zip(last_job_ids, upstream_gradient_batch):
            job_dir = current_job_directory(experiment_dir, job_id)
            adjoint_file_path = job_dir.joinpath(self.upstream_gradient_file_name)
            write_to_csv(adjoint_file_path, np.atleast_2d(grad_y_objective))

        # evaluate the adjoint model
        gradient_response = self.gradient_interface.evaluate(samples)["mean"]

        return response, gradient_response


class ProvidedGradient(GradientHandler):
    """Gradients, directly callable and provided in the forward model.

    Examples are analytical model gradients or gradients provided by
    automated differentiation.
    """

    def __init__(self, model_interface, _get_model_output_fun):
        """Initialize a CallableGrad object.

        Args:
            model_interface (obj): Interface of the associated model.
            _get_model_output_fun (obj): Function that returns the model output.
                                         The type of function is configure beforehand
                                         and depends on the gradient interface
        """
        super().__init__(model_interface)
        self._get_model_output = _get_model_output_fun

    @classmethod
    def from_config_create_grad_handler(cls, grad_obj_name, model_interface, config):
        """Create the gradient object form the problem description.

        Args:
            grad_obj_name (str): Name of the gradient object.
            config (dict): Dictionary with problem description.
            model_interface (obj): Interface of the associated model.
        """
        grad_options = config[grad_obj_name]

        # configure an optional gradient interface
        gradient_interface_name = grad_options.get("gradient_interface_name")
        if gradient_interface_name:
            gradient_interface = from_config_create_interface(gradient_interface_name, config)
        else:
            gradient_interface = None

        _logger.debug(
            "Using the callable/provided gradient from the forward model.",
        )

        # decide which model output function to use
        if gradient_interface:

            def _get_model_output_fun(samples, evaluate_fun):
                return cls._get_output_with_gradient_interface(
                    gradient_interface, samples, evaluate_fun
                )

        else:
            _get_model_output_fun = cls._get_output_without_gradient_interface

        return cls(model_interface, _get_model_output_fun)

    @classmethod
    def _get_output_with_gradient_interface(cls, gradient_interface, samples, evaluate_fun):
        """Evaluate model and its gradients using a gradient interface.

        Args:
            gradient_interface (obj): Interface object for the gradient evaluation.
            samples (np.array): Input samples at which model should be evaluated.
            evaluate_fun (function): Evaluation function that runs the underlying model

        Returns:
            response (np.array): Array with model response for given input samples
            gradient_response_batch (np.array): Array with row-wise model/objective fun
                                                gradients for given samples.
        """
        response = evaluate_fun(samples)["mean"]
        gradient_response_batch = gradient_interface.evaluate(samples)["mean"]
        return response, gradient_response_batch

    @classmethod
    def _get_output_without_gradient_interface(cls, samples, evaluate_fun):
        """Evaluate directly callable model gradients, and model itself.

        Args:
            samples (np.array): Input samples at which model should be evaluated.
            evaluate_fun (function): Evaluation function that runs the underlying model

        Returns:
            response (np.array): Array with model response for given input samples
            gradient_response_batch (np.array): Array with row-wise model/objective fun
                                                gradients for given samples.
        """
        output = evaluate_fun(samples)
        response = output["mean"]
        gradient_response_batch = output["gradient"]
        return response, gradient_response_batch

    def evaluate_and_gradient(self, samples, evaluate_fun, upstream_gradient_fun=None):
        """Evaluate directly callable model gradients, and model itself.

           samples (np.array): Current samples at which model should be evaluated.
           evaluate_fun (obj): Evaluation function that runs the underlying model for the samples
           upstream_gradient_fun (optional, obj): An optional gradient function of an objective
                                                  function which is dependent on the samples and the
                                                  underlying model response (which can be evaluated
                                                  using the evaluate_fun)

        Returns:
            response (np.array): Array with model response for given input samples
            gradient_response (np.array): Array with row-wise model/objective fun gradients for
                                       given samples.
        """
        response, gradient_response_batch = self._get_model_output(samples, evaluate_fun)

        # use chain rule to calculate the upstream gradient in presents of an objective fun
        if upstream_gradient_fun:
            gradient_response_batch = GradientHandler.calculate_downstream_gradient_with_chain_rule(
                upstream_gradient_fun, samples, response, gradient_response_batch
            )

        return response, gradient_response_batch


def prepare_downstream_gradient_fun(
    eval_output_fun, partial_grad_evaluate_fun, upstream_gradient_fun
):
    r"""Prepare the downstream gradient function.

    The method is only necessary for models sitting in between two model levels.
    Such models can receive an upstream gradient function from the superior/upstream
    model level or objective function / functional. They also have to provide their
    own partial gradient function and multiply it with the upstream gradient and remove
    the dependencies on quantities the downstream model is not aware of. The multiplied
    gradient (following chain rule), after all dependencies are removed is called the
    downstream gradient.

    We assume the following arguments for the involved functions, with :math:`x` being the model
    input samples, :math:`y` the output of the sub-model, and :math:`l` the output of the current
    model in the current module at hand:

    - **downstream_eval_fun**: :math:`y = f_{\text{down.eval}}(x)`, not explicitly needed here.
                                Just for completeness to explain the relationship of :math:`x` and
                                :math:`y`.
    - **eval_output_fun**: :math:`l = f_{\text{eval}}(y)`, evaluation method of the current
                           model, containing sub-models, that implement the subordinate eval. fun.
                           when the output of the sub-model is given.
    - **partial_grad_evaluate_fun**:

    .. math::
        \mathbf{\delta}f_{\text{eval}} = \frac{\partial f_{\text{eval}}}{\partial y}(x, y)

    Partial derivative function of current model, w.r.t. the sub-model output

    - **upstream function**: :math:`o = f_{\text{up}}(x, l)`, upstream function, which is
                              handed down to the current model. The upstream function is not
                              directly needed here, only its partial derivative below.
                              We assume that the superior function and its derivative are dependent
                              on the model input :math:`x`
                              and the output of the current model: :math:`l`. (see **eval_fun**).
                              We furthermore assume that the upstream function is `handed-down`
                              from a model, one hierarchy level above, or provided manually in the
                              gradient function.
    - **upstream gradient**:

    .. math::
        \mathbf{\delta}f_{\text{up}} = \frac{\partial f_{\text{up}}}{\partial l}(x, l)

    Partial derivative of the upstream function w.r.t. the output :math:`l` of the current
    model. Note that this function is still dependent on the model output :math:`l`!

    - **downstream_gradient_fun**:

    .. math::
        \mathbf{d}f_{\text{down}} &= \frac{\partial f_{\text{up}}}{\partial y}(x, y)\\
                                 &= \frac{\partial f_{\text{up}}}{\partial l}(x, y)\\
                                 & \cdot \frac{\partial l}{\partial y}(x, y)

    The gradient of the upstream function w.r.t. the sub-model output
    :math:`y` is now the downstream gradient. Also note, that we now resolve
    :math:`l = f_{\text{eval}}(x, y)`, such that the updated
    gradient of the upstream function has the arguments :math:`x, y`.
    In other words: the input and output of the sub-model, when `handed-down`.

    Args:
        eval_fun (obj): Evaluation function of current model for given sub-model output:
                        :math:`l = f_{\text{eval}}(y)`
        partial_grad_evaluate_fun (obj): Partial derivative of the evaluation function of the
                                         current model w.r.t. the output of its sub-model.
                                         :math:`\mathbf{\delta}f_{\text{eval}}`
        upstream_gradient_fun (obj): Partial derivative function of a potentially externally
                                     provided upstream function w.r.t. the output of the current
                                     model:

        ..math::
            \mathbf{\delta}f_{\text{obj}} = \frac{\partial f_{\text{obj}}}{\partial l}(x, l)

    Returns:
        downstream_gradient_fun (obj): The updated gradient of the objective function w.r.t. the
                                       output :math:`l` of the sub-model. The function is only
                                       dependent on the queens input :math:`x` and the sub-model
                                       output :math:`y`:
                                       :math:`\mathbf{d}f_{\text{obj}} = \
                                       \frac{\partial f_{\text{obj}}}{\partial y}(x, y)\
                                       =\frac{\partial f_{\text{obj}}}{\partial l}(x, y)\
                                       \cdot \frac{\partial l}{\partial y}(x, y)`:
    """
    if upstream_gradient_fun:

        def compose_downstream_gradient_fun(samples, sub_model_output):
            """Help to generate new downstream gradient."""
            grad_out = (upstream_gradient_fun(samples, eval_output_fun(sub_model_output))).reshape(
                1, -1
            ) * (partial_grad_evaluate_fun(samples, sub_model_output)).reshape(1, -1)
            return grad_out

        downstream_gradient_fun = compose_downstream_gradient_fun
    else:
        downstream_gradient_fun = partial_grad_evaluate_fun

    return downstream_gradient_fun
