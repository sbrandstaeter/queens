"""
Deterministic optimization

Levenberg Marquardt iterator in the style of baci "gen_inv_analysis"

parts of this class are boldly stolen from optimization_iterator
we need to take control in the details, although it is less flexible it is far more simple

"""
import time

import numpy as np

from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.fd_jacobian import fd_jacobian
from pqueens.utils.process_outputs import write_results


class BaciLMIterator(Iterator):
    """
    Iterator for deterministic optimization problems

    Attributes:
        initial_guess (np.array): initial guess, i.e. start point of
                                  optimization
        max_feval (int) : maximal number of Levenberg-Marquardt iterations
        result_description (dict):  Description of desired
                                    post-processing
    """

    def __init__(
        self,
        global_settings,
        initial_guess,
        jac_rel_step,
        jac_abs_step,
        init_reg,
        update_reg,
        tolerance,
        max_feval,
        model,
        result_description,
        verbose_output,
    ):
        super().__init__(model, global_settings)

        self.initial_guess = initial_guess
        self.param_current = initial_guess
        self.param_old = initial_guess
        self.jac_rel_step = jac_rel_step
        self.max_feval = max_feval
        self.result_description = result_description

        self.jac_abs_step = jac_abs_step
        self.reg_param = init_reg
        self.update_reg = update_reg
        self.tolerance = tolerance

        self.eval_jacobian = False

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
            "Baci LM Iterator for experiment: {0}".format(
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

        initial_guess = np.array(method_options.get('initial_guess'), dtype=float)

        max_feval = method_options.get('max_feval', 1)

        jac_rel_step = method_options.get('jac_rel_step', 1e-4)
        jac_abs_step = method_options.get('jac_abs_step', 0.0)

        init_reg = method_options.get('init_reg', 1.0)
        update_reg = method_options.get('update_reg', 'grad')

        tolerance = method_options.get('convergence_tolerance', 1e-6)

        verbose_output = method_options.get('verbose_output', False)

        # initialize iterator
        return cls(
            global_settings=global_settings,
            initial_guess=initial_guess,
            jac_rel_step=jac_rel_step,
            jac_abs_step=jac_abs_step,
            init_reg=init_reg,
            update_reg=update_reg,
            tolerance=tolerance,
            max_feval=max_feval,
            model=model,
            result_description=result_description,
            verbose_output=verbose_output,
        )

    def eval_model(self):
        """ Evaluate model at current point. """

        result_dict = self.model.evaluate()
        return result_dict

    def residual(self, x0):
        """
        Evaluate objective function at x0.
        """

        x_batch, _ = self.get_positions_raw_2pointperturb(x0)

        precalculated = self.model.check_for_precalculated_response_of_sample_batch(x_batch)

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
        For BACI LM we can restrict to "2-point"
        """

        positions, delta_positions = self.get_positions_raw_2pointperturb(x0)

        precalculated = self.model.check_for_precalculated_response_of_sample_batch(positions)

        # try to recover if response was not found to be precalculated
        if not precalculated:
            self.model.update_model_from_sample_batch(positions)
            self.eval_model()

        f_batch = self.model.response['mean']

        f0 = f_batch[0]  # first entry corresponds to f(x0)
        f_perturbed = np.delete(f_batch, 0, 0)  # delete the first entry

        J = fd_jacobian(f0, f_perturbed, delta_positions, True, '2-point')
        # sanity checks:
        # in the case of LSQ, the number of residuals needs to be
        # greater or equal to the number of parameters to be fitted
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

        print("Initialize BACI Levenberg-Marquardt run.")

    def core_run(self):
        """
        Core run of Optimization iterator
        """
        resnorm = float
        resnorm_o = float

        gradnorm = float
        gradnorm_o = float

        converged = False

        i = 0

        while not converged:

            if i > self.max_feval:
                converged = True
                break

            J = self.jacobian(self.param_current)
            r = self.residual(self.param_current)

            JTJ = (J.T).dot(J)
            JTr = (J.T).dot(r)

            resnorm_o = resnorm
            resnorm = np.linalg.norm(r)

            gradnorm_o = gradnorm
            gradnorm = np.linalg.norm((J.T).dot(r))

            if i == 0:
                resnorm_o = resnorm
                gradnorm_o = gradnorm
                self.lowesterror = resnorm

            #save most optimal step in resnorm
            if resnorm < self.lowesterror:
                self.lowesterror = resnorm
                self.param_opt = self.param_current
                self.iter_opt = i

            # compute update step
            param_delta = -(np.linalg.inv(JTJ + self.reg_param * np.diag(np.diag(JTJ))).dot(JTr))

            # update param for next step
            self.param_old = self.param_current
            self.param_current += param_delta

            print(
                f"iteration: {i} reg_param: {self.reg_param} current_parameters: {self.param_current}"
            )

            # update reg_param and check for tolerance
            if self.update_reg == 'res':
                if resnorm < self.tolerance:
                    converged = True
                    break
                else:
                    self.reg_param = self.reg_param * resnorm / resnorm_o
            elif self.update_reg == 'grad':
                if gradnorm < self.tolerance:
                    converged = True
                    break
                else:
                    if resnorm < resnorm_o:
                        self.reg_param = self.reg_param * gradnorm / gradnorm_o
            else:
                raise ValueError('update_reg unknown')
            i += 1

        self.solution = self.param_opt

    def post_run(self):
        """ Analyze the resulting optimum. """

        print(f"The optimum:\t{self.solution} occured in iteration #{self.iter_opt}.")

        if self.result_description:
            if self.result_description["write_results"]:
                write_results(
                    self.solution,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

    def get_positions_raw_2pointperturb(self, x0):

        delta_positions = self.jac_abs_step + self.jac_rel_step * x0

        positions = np.zeros((x0.size + 1, x0.size))
        positions[0] = x0

        for i in range(x0.size):
            positions[i + 1] = x0
            positions[i + 1][i] += delta_positions[i]

        delta_positions.shape = (delta_positions.size, 1)

        return positions, delta_positions
