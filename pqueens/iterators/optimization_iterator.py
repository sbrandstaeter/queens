"""
Deterministic optimization toolbox

based on the scipy.optimize optimization toolbox [1]

References:
    [1]: https://docs.scipy.org/doc/scipy/reference/optimize.html

@author: Sebastian Brandstaeter
"""
import time

import numpy as np
import scipy.optimize

from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.fd_jacobian import compute_step_with_bounds
from pqueens.utils.fd_jacobian import get_positions
from pqueens.utils.fd_jacobian import fd_jacobian
from pqueens.utils.process_outputs import write_results


class OptimizationIterator(Iterator):
    """
    Iterator for deterministic optimization problems

    Attributes:
        initial_guess (np.array): initial guess, i.e. start point of
                                  optimization
        max_feval (int) : maximal number of function evaluations
        result_description (dict):  Description of desired
                                    post-processing
    """

    def __init__(
        self,
        algorithm,
        bounds,
        constraints,
        global_settings,
        initial_guess,
        jac_method,
        jac_rel_step,
        max_feval,
        model,
        result_description,
        verbose_output,
    ):
        super().__init__(model, global_settings)

        self.algorithm = algorithm
        self.bounds = bounds
        self.cons = constraints
        self.initial_guess = initial_guess
        self.jac_method = jac_method
        self.jac_rel_step = jac_rel_step
        self.max_feval = max_feval
        self.result_description = result_description

        self.eval_jacobian = False
        if self.algorithm in ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'LSQ']:
            self.eval_jacobian = True

        self.verbose_output = verbose_output

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """
        Create Optimization iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: OptimizationIterator object
        """

        print(
            "Optimization Iterator for experiment: {0}".format(
                config.get('global_settings').get('experiment_name')
            )
        )
        if iterator_name is None:
            method_options = config['method']['method_options']
        else:
            method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = Model.from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        initial_guess = np.array(method_options['initial_guess'])

        bounds = method_options.get("bounds", None)

        if bounds is None:
            bounds = [(-np.inf, np.inf)] * initial_guess.shape[0]

        constraints_dict = method_options.get('constraints', None)

        constraints = list()
        if constraints_dict:
            for _, value in constraints_dict.items():
                # evaluate string of lambda function into real lambda function
                value['fun'] = eval(value['fun'])
                constraints.append(value)

        max_feval = method_options.get('max_feval', None)
        algorithm = method_options.get('algorithm', 'L-BFGS-B')
        algorithm = algorithm.upper()

        jac_method = method_options.get('jac_method', '2-point')
        jac_rel_step = method_options.get('jac_rel_step', None)

        verbose_output = method_options.get('verbose_output', False)

        # initialize objective function
        return cls(
            algorithm=algorithm,
            bounds=bounds,
            constraints=constraints,
            global_settings=global_settings,
            initial_guess=initial_guess,
            jac_method=jac_method,
            jac_rel_step=jac_rel_step,
            max_feval=max_feval,
            model=model,
            result_description=result_description,
            verbose_output=verbose_output,
        )

    def eval_model(self):
        """ Evaluate model at current point. """

        result_dict = self.model.evaluate()
        return result_dict

    def objective_function(self, x0):
        """
        Evaluate objective function at x0.
        """

        if self.eval_jacobian:
            x_batch, _ = get_positions(
                x0, method=self.jac_method, rel_step=self.jac_rel_step, bounds=self.bounds
            )
            precalculated = self.model.check_for_precalculated_response_of_sample_batch(x_batch)
        else:
            x_batch = np.atleast_2d(x0)
            precalculated = False

        # try to recover if response was not found to be precalculated
        if not precalculated:
            self.model.update_model_from_sample_batch(x_batch)
            self.eval_model()

        # model response should now correspond to objective function evaluated at positions
        f_batch = np.atleast_1d(np.squeeze(self.model.response['mean']))

        # first entry corresponds to f(x0)
        f0 = f_batch[0]

        return f0

    def jacobian(self, x0):
        """
        Evaluate Jacobian of objective function at x0.
        """

        positions, delta_positions = get_positions(
            x0, method=self.jac_method, rel_step=self.jac_rel_step, bounds=self.bounds
        )
        _, use_one_sided = compute_step_with_bounds(
            x0, method=self.jac_method, rel_step=self.jac_rel_step, bounds=self.bounds
        )

        precalculated = self.model.check_for_precalculated_response_of_sample_batch(positions)

        # try to recover if response was not found to be precalculated
        if not precalculated:
            self.model.update_model_from_sample_batch(positions)
            self.eval_model()

        # model response should now correspond to objective function evaluated at positions
        f_batch = self.model.response['mean']

        f0 = f_batch[0]  # first entry corresponds to f(x0)
        f_perturbed = np.delete(f_batch, 0, 0)  # delete the first entry

        J = fd_jacobian(f0, f_perturbed, delta_positions, use_one_sided, method=self.jac_method)
        # sanity checks:
        # in the case of LSQ, the number of residuals needs to be
        # greater or equal to the number of parameters to be fitted
        if self.algorithm == 'LSQ' and J.ndim == 2:
            num_res, num_par = J.shape
            if num_res < num_par:
                raise ValueError(
                    f"Number of residuals (={num_res}) has to be greater or equal to"
                    f" number of parameters (={num_par})."
                    f" You have {num_res}<{num_par}."
                )
        return J

    def initialize_run(self):
        """ Get initial guess. """

        print("Initialize Optimization run.")

    def core_run(self):
        """
        Core run of Optimization iterator
        """

        print('Welcome to Optimization core run.')
        start = time.time()
        # nonlinear least squares with bounds using Jacobian
        if self.algorithm == 'LSQ':
            self.solution = scipy.optimize.least_squares(
                self.objective_function,
                self.initial_guess,
                jac=self.jacobian,
                bounds=self.bounds,
                max_nfev=self.max_feval,
                verbose=int(self.verbose_output),
            )
        # minimization with bounds using Jacobian
        elif self.algorithm in {'L-BFGS-B', 'TNC'}:
            self.solution = scipy.optimize.minimize(
                self.objective_function,
                self.initial_guess,
                method=self.algorithm,
                jac=self.jacobian,
                bounds=self.bounds,
                options={'maxiter': int(1e4), 'disp': self.verbose_output},
            )
        # Constrained Optimimization BY Linear Approximation:
        # minimization with constraints without Jacobian
        elif self.algorithm in {'COBYLA'}:
            self.solution = scipy.optimize.minimize(
                self.objective_function,
                self.initial_guess,
                method=self.algorithm,
                constraints=self.cons,
                options={'disp': self.verbose_output},
            )
        # Sequential Least SQuares Programming:
        # minimization with bounds and constraints using Jacobian
        elif self.algorithm in {'SLSQP'}:
            self.solution = scipy.optimize.minimize(
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
            self.solution = scipy.optimize.minimize(
                self.objective_function,
                self.initial_guess,
                method=self.algorithm,
                options={'disp': self.verbose_output},
            )
        # minimization (unconstrained, unbounded) using Jacobian
        elif self.algorithm in {'CG', 'BFGS'}:
            self.solution = scipy.optimize.minimize(
                self.objective_function,
                self.initial_guess,
                method=self.algorithm,
                jac=self.jacobian,
                options={'disp': self.verbose_output},
            )
        end = time.time()
        print(f"Optimization took {end-start} seconds.")

    def post_run(self):
        """ Analyze the resulting optimum. """

        print(f"The optimum:\n\t{self.solution.x}")
        if self.algorithm == 'LSQ':
            print(f"Optimality:\n\t{self.solution.optimality}")
            print(f"Cost:\n\t{self.solution.cost}")

        if self.result_description:
            if self.result_description["write_results"]:
                write_results(
                    self.solution,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )
