"""Deterministic optimization toolbox."""
import logging
import time

import numpy as np
from scipy.optimize import least_squares, minimize

from queens.iterators.iterator import Iterator
from queens.utils.fd_jacobian import fd_jacobian, get_positions
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class OptimizationIterator(Iterator):
    """Iterator for deterministic optimization problems.

    Based on the *scipy.optimize* optimization toolbox [1].

    References:
        [1]: https://docs.scipy.org/doc/scipy/reference/optimize.html

    Attributes:
        algorithm (str): String that defines the optimization algorithm to be used:

                         - CG: Conjugate gradient optimization (unconstrained), using Jacobian
                         - BFGS: Broyden–Fletcher–Goldfarb–Shanno algorithm (quasi-Newton) for
                                optimization (iterative method for unconstrained
                                nonlinear optimization), using Jacobian
                         - L-BFGS-B: Limited memory Broyden–Fletcher–Goldfarb–Shanno algorithm
                                     with box constraints (for large number of variables)
                         - TNC: Truncated Newton method (Hessian free) for nonlinear
                                optimization with bounds involving a large number of variables.
                                Jacobian necessary
                         - SLSQP: Sequential Least Squares Programming minimization with bounds
                                  and constraints using Jacobian
                         - LSQ: Nonlinear least squares with bounds using Jacobian
                         - COBYLA: Constrained Optimization BY Linear Approximation
                                   (no Jacobian)
                         - NELDER-MEAD: Downhill-simplex search method
                                        (unconstrained, unbounded)
                                        without the need for a Jacobian
                         - POWELL: Powell's conjugate direction method (unconstrained) without
                                   the need for a Jacobian. Minimizes the function by a
                                   bidirectional search along each search vector
        bounds (sequence, Bounds): Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP,
                                   Powell, and trust-constr methods. See SciPy documentation on
                                   how to specify bounds.
        cons (np.array): Nonlinear constraints for the optimization.
                         Only for COBYLA, SLSQP and trust-constr
                         (see SciPy documentation for details)
        initial_guess (np.array): Initial guess, i.e. start point of
                                  optimization.
        jac_method (str): Method to calculate a finite difference based approximation of the
                          Jacobian matrix:

                          - '2-point': a one-sided scheme by definition
                          - '3-point': more exact but needs twice as many function evaluations
        jac_rel_step (array_like): Relative step size to use for finite difference approximation
                                   of Jacobian matrix. If None (default) then it is selected
                                   automatically. (see SciPy documentation for details)
        max_feval (int): Maximum number of function evaluations.
        result_description (dict): Description of desired post-processing.
        verbose_output (int): Integer encoding which kind of verbose information should be
                              printed by the optimizers.
        precalculated_positions (dict): Dictionary containing precalculated positions and
                                        corresponding model responses.
        solution (np.array): Solution obtained from the optimization process.

    Returns:
        OptimizationIterator (obj): Instance of the OptimizationIterator
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        initial_guess,
        result_description,
        verbose_output=False,
        bounds=None,
        constraints=None,
        max_feval=None,
        algorithm='L-BFGS-B',
        jac_method='2-point',
        jac_rel_step=None,
    ):
        """Initialize an OptimizationIterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            initial_guess (array like): initial position at which the optimization starts
            result_description (dict): Description of desired post-processing.
            verbose_output (int): Integer encoding which kind of verbose information should be
                                  printed by the optimizers.
            bounds (sequence, Bounds): Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP,
                                       Powell, and trust-constr methods. See SciPy documentation on
                                       how to specify bounds.
            constraints (np.array): Nonlinear constraints for the optimization.
                                    Only for COBYLA, SLSQP and trust-constr
                                    (see SciPy documentation for details)
            max_feval (int): Maximum number of function evaluations.
            algorithm (str): String that defines the optimization algorithm to be used:

                             - CG: Conjugate gradient optimization (unconstrained), using Jacobian
                             - BFGS: Broyden–Fletcher–Goldfarb–Shanno algorithm (quasi-Newton) for
                                    optimization (iterative method for unconstrained
                                    nonlinear optimization), using Jacobian
                             - L-BFGS-B: Limited memory Broyden–Fletcher–Goldfarb–Shanno algorithm
                                         with box constraints (for large number of variables)
                             - TNC: Truncated Newton method (Hessian free) for nonlinear
                                    optimization with bounds involving a large number of variables.
                                    Jacobian necessary
                             - SLSQP: Sequential Least Squares Programming minimization with bounds
                                      and constraints using Jacobian
                             - LSQ: Nonlinear least squares with bounds using Jacobian
                             - COBYLA: Constrained Optimization BY Linear Approximation
                                       (no Jacobian)
                             - NELDER-MEAD: Downhill-simplex search method
                                            (unconstrained, unbounded)
                                            without the need for a Jacobian
                             - POWELL: Powell's conjugate direction method (unconstrained) without
                                       the need for a Jacobian. Minimizes the function by a
                                       bidirectional search along each search vector
            jac_method (str): Method to calculate a finite difference based approximation of the
                              Jacobian matrix:

                              - '2-point': a one-sided scheme by definition
                              - '3-point': more exact but needs twice as many function evaluations
            jac_rel_step (array_like): Relative step size to use for finite difference approximation
                                       of Jacobian matrix. If None (default) then it is selected
                                       automatically. (see SciPy documentation for details)
        """
        super().__init__(model, parameters)
        _logger.info("Optimization Iterator for experiment: %s", self.experiment_name)

        initial_guess = np.atleast_1d(np.array(initial_guess))

        constraints_list = []
        if constraints:
            for value in constraints.values():
                # evaluate string of lambda function into real lambda function
                value['fun'] = eval(value['fun'])  # pylint: disable=eval-used
                constraints_list.append(value)

        algorithm = algorithm.upper()

        self.algorithm = algorithm
        self.bounds = bounds
        self.cons = constraints_list
        self.initial_guess = initial_guess
        self.jac_method = jac_method
        self.jac_rel_step = jac_rel_step
        self.max_feval = max_feval
        self.result_description = result_description
        self.verbose_output = verbose_output
        self.precalculated_positions = {'position': [], 'output': []}
        self.solution = None

    def objective_function(self, x_vec):
        """Evaluate objective function at *x_vec*.

        Args:
            x_vec (np.array): Input vector for model/objective function. The variable *x_vec*
                              corresponds to the parameters that should be calibrated in an
                              inverse problem.

        Returns:
            f_value (float): Response of objective function or model
        """
        f_value = self.eval_model(x_vec)

        parameter_list = self.parameters.parameters_keys
        _logger.info("The intermediate, iterated parameters %s are:\n\t%s", parameter_list, x_vec)

        return f_value

    def jacobian(self, x0):
        """Evaluate Jacobian of objective function at *x0*.

        Args:
            x0 (np.array): position to evaluate Jacobian at
        Returns:
            jacobian_matrix (np.array): Jacobian matrix evaluated at *x0*
        """
        additional_positions, delta_positions, use_one_sided = get_positions(
            x0, method=self.jac_method, rel_step=self.jac_rel_step, bounds=self.bounds
        )

        # model response should now correspond to objective function evaluated at positions
        positions = np.vstack((x0, additional_positions))
        f_batch = self.eval_model(positions)

        f0 = f_batch[0].reshape(-1)  # first entry corresponds to f(x0)
        f_perturbed = f_batch[1:].reshape(-1, f0.size)

        jacobian_matrix = fd_jacobian(
            f0, f_perturbed, delta_positions, use_one_sided, method=self.jac_method
        )
        # sanity checks:
        # in the case of LSQ, the number of residuals needs to be
        # greater or equal to the number of parameters to be fitted
        if self.algorithm == 'LSQ' and jacobian_matrix.ndim == 2:
            num_res, num_par = jacobian_matrix.shape
            if num_res < num_par:
                raise ValueError(
                    f"Number of residuals (={num_res}) has to be greater or equal to"
                    f" number of parameters (={num_par})."
                    f" You have {num_res}<{num_par}."
                )
        return jacobian_matrix

    def pre_run(self):
        """Pre run of Optimization iterator."""
        _logger.info("Initialize Optimization run.")

    def core_run(self):
        """Core run of Optimization iterator."""
        _logger.info('Welcome to Optimization core run.')
        start = time.time()
        # nonlinear least squares with bounds using Jacobian
        if self.algorithm == 'LSQ':
            self.solution = least_squares(
                self.objective_function,
                self.initial_guess,
                jac=self.jacobian,
                bounds=self.bounds,
                max_nfev=self.max_feval,
                verbose=int(self.verbose_output),
            )
        # minimization with bounds using Jacobian
        elif self.algorithm in {'L-BFGS-B', 'TNC'}:
            self.solution = minimize(
                self.objective_function,
                self.initial_guess,
                method=self.algorithm,
                jac=self.jacobian,
                bounds=self.bounds,
                options={'maxiter': int(1e4), 'disp': self.verbose_output},
            )
        # Constrained Optimization BY Linear Approximation:
        # minimization with constraints without Jacobian
        elif self.algorithm in {'COBYLA'}:
            self.solution = minimize(
                self.objective_function,
                self.initial_guess,
                method=self.algorithm,
                constraints=self.cons,
                options={'disp': self.verbose_output},
            )
        # Sequential Least SQuares Programming:
        # minimization with bounds and constraints using Jacobian
        elif self.algorithm in {'SLSQP'}:
            self.solution = minimize(
                self.objective_function,
                self.initial_guess,
                method=self.algorithm,
                jac=self.jacobian,
                bounds=self.bounds,
                constraints=self.cons,
                options={'disp': self.verbose_output},
            )
        # minimization (unconstrained, unbounded) without Jacobian
        elif self.algorithm in {'NELDER-MEAD', 'POWELL'}:
            self.solution = minimize(
                self.objective_function,
                self.initial_guess,
                method=self.algorithm,
                options={'disp': self.verbose_output},
            )
        # minimization (unconstrained, unbounded) using Jacobian
        elif self.algorithm in {'CG', 'BFGS'}:
            self.solution = minimize(
                self.objective_function,
                self.initial_guess,
                method=self.algorithm,
                jac=self.jacobian,
                options={'disp': self.verbose_output},
            )
        end = time.time()
        _logger.info("Optimization took %E seconds.", end - start)

    def post_run(self):
        """Analyze the resulting optimum."""
        if self.algorithm == 'LM':
            parameter_list = self.parameters.parameters_keys()
            _logger.info(
                "The optimum of the parameters %s is:\n\t%s", *parameter_list, self.solution[0]
            )
        else:
            _logger.info("The optimum:\n\t%s", self.solution.x)
            if self.algorithm == 'LSQ':
                _logger.info("Optimality:\n\t%s", self.solution.optimality)
                _logger.info("Cost:\n\t%s", self.solution.cost)

        if self.result_description:
            if self.result_description["write_results"]:
                write_results(self.solution, self.output_dir, self.experiment_name)

    def eval_model(self, positions):
        """Evaluate model at defined positions.

        Args:
            positions (np.ndarray): Positions at which the model is evaluated

        Returns:
            f_batch (np.ndarray): Model response
        """
        positions = positions.reshape(-1, self.parameters.num_parameters)
        f_batch = []
        for position in positions:
            precalculated_output = self.check_precalculated(position)
            if precalculated_output is not None:
                f_batch.append(precalculated_output)
            else:
                f_batch.append(self.model.evaluate(position.reshape(1, -1))['result'].reshape(-1))
                self.precalculated_positions['position'].append(position)
                self.precalculated_positions['output'].append(f_batch[-1])
        f_batch = np.array(f_batch).squeeze()
        return f_batch

    def check_precalculated(self, position):
        """Check if the model was already evaluated at defined position.

        Args:
            position (np.ndarray): Position at which the model should be evaluated

        Returns:
            np.ndarray: Precalculated model response or *None*
        """
        for i, precalculated_position in enumerate(self.precalculated_positions['position']):
            if np.equal(position, precalculated_position).all():
                return self.precalculated_positions['output'][i]
        return None
