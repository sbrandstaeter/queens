"""Levenberg Marquardt iterator."""
import logging
import os

import numpy as np
import pandas as pd
import plotly.express as px

from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils.fd_jacobian import fd_jacobian

_logger = logging.getLogger(__name__)


class BaciLMIterator(Iterator):
    """Iterator for deterministic optimization problems.

    Levenberg Marquardt iterator in the style of baci "gen_inv_analysis"

    parts of this class are boldly stolen from optimization_iterator
    we need to take control in the details, although it is less flexible it is far more simple
    """

    def __init__(
        self,
        global_settings,
        initial_guess,
        bounds,
        havebounds,
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
        self.bounds = bounds
        self.havebounds = havebounds
        self.param_current = initial_guess
        self.jac_rel_step = jac_rel_step
        self.max_feval = max_feval
        self.result_description = result_description

        self.jac_abs_step = jac_abs_step
        self.reg_param = init_reg
        self.init_reg = init_reg
        self.update_reg = update_reg
        self.tolerance = tolerance

        self.verbose_output = verbose_output
        self.iter_opt = 0

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create Levenberg Marquardt iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: OptimizationIterator object
        """
        _logger.info(
            "Baci LM Iterator for experiment: %s",
            config.get('global_settings').get('experiment_name'),
        )

        method_options = config[iterator_name]
        if model is None:
            model_name = method_options['model_name']
            model = from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        initial_guess = np.array(method_options.get('initial_guess'), dtype=float)

        bounds = method_options.get("bounds", None)
        havebounds = True
        if bounds is None:
            havebounds = False

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
            bounds=bounds,
            havebounds=havebounds,
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

    def jacobian_and_residual(self, x0):
        """Evaluate Jacobian and residual of objective function at x0.

        For BACI LM we can restrict to "2-point".

        Args:
            x0 (numpy.ndarray): vector with current parameters

        Returns:
            J (numpy.ndarray): Jacobian Matrix approximation from finite differences
            f0 (numpy.ndarray): residual of objective function at x0
        """
        positions, delta_positions = self.get_positions_raw_2pointperturb(x0)

        self.model.evaluate(positions)

        f = self.model.response['mean']
        f_batch = f[-(len(positions)) :]

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

        return J, f0

    def pre_run(self):
        """Initialize run.

        Print console output and optionally open .csv file for results
        and write header.
        """
        _logger.info("Initialize BACI Levenberg-Marquardt run.")

        # produce .csv file and write header
        if self.result_description:
            if self.result_description["write_results"]:
                df = pd.DataFrame(
                    columns=['iter', 'resnorm', 'gradnorm', 'params', 'delta_params', 'mu'],
                )
                with open(
                    os.path.join(
                        self.global_settings["output_dir"], self.global_settings["experiment_name"]
                    )
                    + '.csv',
                    'w',
                ) as f:
                    df.to_csv(f, sep='\t', index=None)

    def core_run(self):
        """Core run of Levenberg Marquardt iterator."""
        resnorm = np.inf
        gradnorm = np.inf

        converged = False

        i = 0

        # Levenberg Marquardt iterations
        while not converged:

            if i > self.max_feval:
                converged = True
                _logger.info('Maximum number of steps max_feval= %d reached.', self.max_feval)
                break

            _logger.info(
                "iteration: %d reg_param: %s current_parameters: %s",
                i,
                self.reg_param,
                self.param_current,
            )
            jacobian, residual = self.jacobian_and_residual(self.param_current)

            # store terms for repeated useage
            JTJ = (jacobian.T).dot(jacobian)
            JTr = (jacobian.T).dot(residual)

            resnorm_o = resnorm
            resnorm = np.linalg.norm(residual) / np.sqrt(residual.size)

            gradnorm_o = gradnorm
            gradnorm = np.linalg.norm(JTr)

            if i == 0:
                resnorm_o = resnorm
                gradnorm_o = gradnorm
                self.lowesterror = resnorm
                self.param_opt = self.param_current

            # save most optimal step in resnorm
            if resnorm < self.lowesterror:
                self.lowesterror = resnorm
                self.param_opt = self.param_current
                self.iter_opt = i

            # compute update step. Not necessary for last iteration! Should usually be low
            # dimension matrix inversion. We need J and r anyway for convergence check.
            param_delta = np.linalg.solve(JTJ + self.reg_param * np.diag(np.diag(JTJ)), -(JTr))

            # output for iteration. Before update including next step
            self.printstep(i, resnorm, gradnorm, param_delta)

            # evaluate bounds
            if self.havebounds and self.checkbounds(param_delta, i):
                if self.reg_param > 1.0e6 * self.init_reg:
                    _logger.info(
                        'WARNING: STEP #%d IS OUT OF BOUNDS and reg_param is unreasonably '
                        'high. Ending iterations!',
                        i,
                    )
                    break
                else:
                    continue

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
                    if resnorm < resnorm_o and gradnorm < gradnorm_o:
                        self.reg_param = self.reg_param * gradnorm / gradnorm_o
            else:
                raise ValueError('update_reg unknown')
            i += 1

        # store set of parameters which leads to lowest residual as solution
        self.solution = self.param_opt

    def post_run(self):
        """Post run.

        Write solution to console and optionally create .html plot from
        result file.
        """
        _logger.info(f"The optimum:\t{self.solution} occured in iteration #{self.iter_opt}.")
        if self.result_description:
            if self.result_description["plot_results"] and self.result_description["write_results"]:
                data = pd.read_csv(
                    os.path.join(
                        self.global_settings["output_dir"], self.global_settings["experiment_name"]
                    )
                    + '.csv',
                    sep='\t',
                )
                xydata = data['params']
                xydata = xydata.str.extractall(r'([+-]?\d+\.\d*e?[+-]?\d*)')
                xydata = xydata.unstack()
                data = data.drop(columns='params')
                i = 0
                for column in xydata:
                    data[self.parameters.names[i]] = xydata[column].astype(float)
                    i = i + 1

                if i > 2:
                    _logger.warning(
                        'write_results for more than 2 parameters not implemented, '
                        'because we are limited to 3 dimensions. '
                        'You have: %d. Plotting is skipped.',
                        i,
                    )
                    return
                elif i == 2:
                    fig = px.line_3d(
                        data,
                        x=self.parameters.names[0],
                        y=self.parameters.names[1],
                        z='resnorm',
                        hover_data=[
                            'iter',
                            'resnorm',
                            'gradnorm',
                            'delta_params',
                            'mu',
                            self.parameters.names[0],
                            self.parameters.names[1],
                        ],
                    )
                    fig.update_traces(mode='lines+markers', marker=dict(size=2), line=dict(width=4))
                elif i == 1:
                    fig = px.line(
                        data,
                        x=self.parameters.names[0],
                        y='resnorm',
                        hover_data=[
                            'iter',
                            'resnorm',
                            'gradnorm',
                            'delta_params',
                            'mu',
                            self.parameters.names[0],
                        ],
                    )
                    fig.update_traces(mode='lines+markers', marker=dict(size=7), line=dict(width=3))
                else:
                    raise ValueError('You shouldn\'t be here without parameters.')

                fig.write_html(
                    os.path.join(
                        self.global_settings["output_dir"], self.global_settings["experiment_name"]
                    )
                    + '.html'
                )

    def get_positions_raw_2pointperturb(self, x0):
        """Get parameter sets for objective function evaluations.

        Args:
            x0 (numpy.ndarray): vector with current parameters

        Returns:
            positions (numpy.ndarray): parameter batch for function evaluation
            delta_positions (numpy.ndarray): parameter perturbations for finite difference scheme
        """
        delta_positions = self.jac_abs_step + self.jac_rel_step * np.abs(x0)
        # if bounds do not allow finite difference step, use opposite direction
        if self.havebounds:
            perturbed = x0 + delta_positions
            for i, current_p in enumerate(perturbed):
                if (current_p < self.bounds[0][i]) or (current_p > self.bounds[1][i]):
                    delta_positions[i] = -delta_positions[i]

        positions = np.zeros((x0.size + 1, x0.size))
        positions[0] = x0

        for i in range(x0.size):
            positions[i + 1] = x0
            positions[i + 1][i] = positions[i + 1][i] + delta_positions[i]

        delta_positions.shape = (delta_positions.size, 1)

        return positions, delta_positions

    def printstep(self, i, resnorm, gradnorm, param_delta):
        """Print iteration data to console and optionally to file.

        Opens file in append mode, so that file is updated frequently.

        Args:
            i (int): iteration number
            resnorm (float): residual norm
            gradnorm (float): gradient norm
            param_delta (numpy.ndarray): parameter step

        Returns:
            None
        """
        # write iteration to file
        if self.result_description:
            if self.result_description["write_results"]:
                with open(
                    os.path.join(
                        self.global_settings["output_dir"], self.global_settings["experiment_name"]
                    )
                    + '.csv',
                    'a',
                ) as f:
                    df = pd.DataFrame(
                        {
                            'iter': i,
                            'resnorm': np.format_float_scientific(resnorm, precision=8),
                            'gradnorm': np.format_float_scientific(gradnorm, precision=8),
                            'params': [np.array2string(self.param_current, precision=8)],
                            'delta_params': [np.array2string(param_delta, precision=8)],
                            'mu': np.format_float_scientific(self.reg_param, precision=8),
                        }
                    )
                    df.to_csv(f, sep='\t', header=None, mode='a', index=None, float_format='%.8f')

    def checkbounds(self, param_delta, i):
        """check if proposed step is in bounds, otherwise double regularization
        and compute new step.

        Args:
            param_delta (numpy.ndarray): parameter step
            i (int): iteration number

        Returns:
            stepisoutside (bool): flag if proposed step is out of bounds
        """
        stepisoutside = False
        nextstep = self.param_current + param_delta
        if np.any(nextstep < self.bounds[0]) or np.any(nextstep > self.bounds[1]):
            stepisoutside = True
            _logger.warning(
                'WARNING: STEP #%d IS OUT OF BOUNDS; double reg_param and compute new iteration.'
                '\n declined step was: %s',
                i,
                nextstep,
            )

            self.reg_param *= 2

        return stepisoutside
