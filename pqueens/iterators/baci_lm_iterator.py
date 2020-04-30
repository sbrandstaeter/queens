import numpy as np
import os
import pandas as pd
import plotly.express as px

from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.fd_jacobian import fd_jacobian


class BaciLMIterator(Iterator):
    """
    Iterator for deterministic optimization problems

    Levenberg Marquardt iterator in the style of baci "gen_inv_analysis"

    parts of this class are boldly stolen from optimization_iterator
    we need to take control in the details, although it is less flexible it is far more simple

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
        Create Levenberg Marquardt iterator from problem description

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

        Args:
            x0 (numpy.ndarray): vector with current parameters

        Returns:
            f0 (numpy.ndarray): current residual vector
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

        Args:
            x0 (numpy.ndarray): vector with current parameters

        Returns:
            J (numpy.ndarray): Jacobian Matrix approximation from finite differences
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
        """
        Print console output and optionally open .csv file for results and write header.

        Args:
            none

        Returns:
            none
        """

        print("Initialize BACI Levenberg-Marquardt run.")

        # produce .csv file and write header
        if self.result_description:
            if self.result_description["write_results"]:
                df = pd.DataFrame(
                    columns=['iter', 'resnorm', 'gradnorm', 'params', 'delta_params', 'mu'],
                )
                f = open(
                    os.path.join(
                        self.global_settings["output_dir"], self.global_settings["experiment_name"]
                    )
                    + '.csv',
                    'w',
                )
                df.to_csv(f, sep='\t', index=None)
                f.close()

    def core_run(self):
        """
        Core run of Levenberg Marquardt iterator

        Args:
            none

        Returns:
            none
        """

        resnorm = float
        resnorm_o = float

        gradnorm = float
        gradnorm_o = float

        converged = False

        i = 0

        # Levenberg Marquardt iterations
        while not converged:

            if i > self.max_feval:
                converged = True
                print(f'Maximum number of steps max_feval= {self.max_feval} reached.')
                break

            J = self.jacobian(self.param_current)
            r = self.residual(self.param_current)

            # store terms for repeated useage
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

            # save most optimal step in resnorm
            if resnorm < self.lowesterror:
                self.lowesterror = resnorm
                self.param_opt = self.param_current
                self.iter_opt = i

            # compute update step. Not necessary for last iteration! Should usually be low
            # dimension matrix inversion. We need J and r anyway for convergence check.
            param_delta = -(np.linalg.inv(JTJ + self.reg_param * np.diag(np.diag(JTJ))).dot(JTr))

            # output for iteration. Before update including next step
            self.printstep(i, resnorm, gradnorm, param_delta)

            # update param for next step
            self.param_current = self.param_current + param_delta

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

        # store set of parameters which leads to lowest residual as solution
        self.solution = self.param_opt

    def post_run(self):
        """
        Write solution to console and optionally create .html plot from result file.

        Args:
        none

        Returns:
        none
        """

        print(f"The optimum:\t{self.solution} occured in iteration #{self.iter_opt}.")
        if self.result_description:
            if self.result_description["plot_results"]:
                data = pd.read_csv(
                    os.path.join(
                        self.global_settings["output_dir"], self.global_settings["experiment_name"]
                    )
                    + '.csv',
                    sep='\t',
                )
                xydata = data['params']
                xydata = xydata.str.extractall('([+-]?\d+\.\d*e?[+-]?\d*)')
                xydata = xydata.unstack()
                data = data.drop(columns='params')
                i = 0
                var_name = [*self.model.variables[0].variables.keys()]
                for column in xydata:
                    data[var_name[i]] = xydata[column]
                    i = i + 1

                if i > 2:
                    print(
                        f'write_results for more than 2 parameters not implemented, '
                        f'because we are limited to 3 dimensions. '
                        f'You have: {i}.'
                    )
                    pass

                fig = px.line_3d(
                    data,
                    x=var_name[0],
                    y=var_name[1],
                    z='resnorm',
                    hover_data=[
                        'iter',
                        'resnorm',
                        'gradnorm',
                        'delta_params',
                        'mu',
                        var_name[0],
                        var_name[1],
                    ],
                )
                fig.update_traces(mode='lines+markers', marker=dict(size=2), line=dict(width=4))
                fig.write_html(
                    os.path.join(
                        self.global_settings["output_dir"], self.global_settings["experiment_name"]
                    )
                    + '.html'
                )
        pass

    def get_positions_raw_2pointperturb(self, x0):

        """
        Get parameter sets for objective function evaluations.

        Args:
        x0 (numpy.ndarray): vector with current parameters

        Returns:
        positions (numpy.ndarray): parameter batch for function evluation
        delta_positions (numpy.array): parameter perturbations for finite difference scheme
        """

        delta_positions = self.jac_abs_step + self.jac_rel_step * x0

        positions = np.zeros((x0.size + 1, x0.size))
        positions[0] = x0

        for i in range(x0.size):
            positions[i + 1] = x0
            positions[i + 1][i] = positions[i + 1][i] + delta_positions[i]

        delta_positions.shape = (delta_positions.size, 1)

        return positions, delta_positions

    def printstep(self, i, resnorm, gradnorm, param_delta):

        """
        Print iteration data to console and optionally to file.
        Opens file in append mode, so that file is updated frequently

        Args:
        i (int): iteration number
        resnorm (float): residual norm
        gradnorm (float): gradient norm
        param_delta (numpy.ndarray): parameter step

        Returns:
        None
        """
        print(
            f"iteration: {i} reg_param: {self.reg_param} current_parameters: {self.param_current}"
        )

        # write iteration to file
        if self.result_description:
            if self.result_description["write_results"]:
                f = open(
                    os.path.join(
                        self.global_settings["output_dir"], self.global_settings["experiment_name"]
                    )
                    + '.csv',
                    'a',
                )
                df = pd.DataFrame(
                    {
                        'iter': i,
                        'resnorm': resnorm,
                        'gradnorm': gradnorm,
                        'params': [self.param_current],
                        'delta_params': [param_delta],
                        'mu': self.reg_param,
                    }
                )
                df.to_csv(f, sep='\t', header=None, mode='a', index=None, float_format='%.6f')
                f.close()
        pass
