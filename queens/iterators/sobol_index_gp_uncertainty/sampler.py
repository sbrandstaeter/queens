"""Helper class for generating Monte-Carlo samples for Sobol indices."""
import logging

import numpy as np
import xarray as xr
from scipy.stats import qmc

_logger = logging.getLogger(__name__)


class Sampler:
    """Sampler class.

    Draw Monte-Carlo samples and generate sample matrices A, B, AB, BA following

    Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
    Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270.
    https://doi.org/10.1016/j.cpc.2009.09.018.

    Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
    John Wiley & Sons, Ltd., 2008.
    https://doi.org/10.1002/9780470725184.

    Attributes:
        number_parameters (int): number of input space dimensions
        number_monte_carlo_samples (int): number of Monte-Carlo samples
        sampling_approach (string): sampling approach (pseudo-random or quasi-random)
        calculate_second_order (bool): true if second-order indices are calculated
        parameters (list): information about distribution of random variables
        parameter_names (list): list of names of input parameters
        seed_monte_carlo (int): seed for random samples
    """

    def __init__(
        self,
        parameter_names,
        number_monte_carlo_samples,
        seed_monte_carlo,
        calculate_second_order,
        parameters,
        sampling_approach,
    ):
        """Initialize.

        Args:
            parameter_names (list): list of names of input parameters
            number_monte_carlo_samples (int): number of Monte-Carlo samples
            seed_monte_carlo (int): seed for random samples
            calculate_second_order (bool): true if second-order indices are calculated
            parameters (list): information about distribution of random variables
            sampling_approach (string): sampling approach (pseudo-random or quasi-random)
        """
        self.parameter_names = parameter_names
        self.number_parameters = len(self.parameter_names)
        self.number_monte_carlo_samples = number_monte_carlo_samples
        self.seed_monte_carlo = seed_monte_carlo
        self.calculate_second_order = calculate_second_order
        self.parameters = parameters
        self.sampling_approach = sampling_approach

    @classmethod
    def from_config_create(cls, method_options, parameter_names, parameters):
        """Create sampler from problem description.

        Args:
            method_options (dict): dictionary with method options
            parameter_names (list): list of parameter names
            parameters (list): information about distribution of random variables

        Returns:
            sampler: Sampler object
        """
        number_monte_carlo_samples = method_options['number_monte_carlo_samples']
        _logger.info('Number of Monte-Carlo samples = %i', number_monte_carlo_samples)

        seed_monte_carlo = method_options.get('seed_monte_carlo', 42)

        calculate_second_order = method_options.get("second_order", False)
        sampling_approach = method_options.get('sampling_approach', 'quasi_random')

        return cls(
            parameter_names=parameter_names,
            number_monte_carlo_samples=number_monte_carlo_samples,
            seed_monte_carlo=seed_monte_carlo,
            calculate_second_order=calculate_second_order,
            parameters=parameters,
            sampling_approach=sampling_approach,
        )

    def sample(self):
        """Generate sample matrices.

        Generate sample matrices A, B, AB, BA following

        Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
        Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
        (1 February 2010): 259–270.
        https://doi.org/10.1016/j.cpc.2009.09.018.

        Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
        John Wiley & Sons, Ltd., 2008.
        https://doi.org/10.1002/9780470725184.

        A:  sample matrix A
        B:  sample matrix B
        AB:  separated from A by only change in X_i; from B by change in X_{~i}
        BA:  separated from B by only change in X_i; from A by change in X_{~i}

        Returns:
            samples (xr.Array): Monte-Carlo samples
        """
        samples = self._init_samples()

        A, B = self._draw_base_samples()
        samples.loc[{"sample_matrix": 'A'}] = A
        samples.loc[{"sample_matrix": 'B'}] = B

        for i, parameter_name in enumerate(self.parameter_names):
            AB = A.copy()
            AB[:, i] = B[:, i].flatten()
            samples.loc[{"sample_matrix": 'AB_' + parameter_name}] = AB
            if self.calculate_second_order:
                BA = B.copy()
                BA[:, i] = A[:, i].flatten()
                samples.loc[{"sample_matrix": 'BA_' + parameter_name}] = BA

        return samples

    def _init_samples(self):
        """Initialize samples data-array.

        Returns:
            samples (xr.DataArray): Monte-Carlo samples (sample_matrix x M x D)
        """
        labels_sample_matrices = self._setup_sample_matrices_labels()
        dimensions = ("monte_carlo", "sample_matrix", "parameter")
        coordinates = {
            "monte_carlo": np.arange(self.number_monte_carlo_samples),
            "sample_matrix": labels_sample_matrices,
            "parameter": self.parameter_names,
        }

        samples = xr.DataArray(
            data=np.empty(
                (
                    self.number_monte_carlo_samples,
                    len(labels_sample_matrices),
                    self.number_parameters,
                )
            ),
            dims=dimensions,
            coords=coordinates,
        )

        return samples

    def _setup_sample_matrices_labels(self):
        """Set up labels of sample matrices.

        For further information on the sample matrices see

        Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
        Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
        (1 February 2010): 259–270.
        https://doi.org/10.1016/j.cpc.2009.09.018.

        Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
        John Wiley & Sons, Ltd., 2008.
        https://doi.org/10.1002/9780470725184.

        Examples:
            ['AB_x1', 'AB_x2', 'AB_x3', 'A', 'B'] (only first and total-order)
            ['AB_x1', 'AB_x2', 'AB_x3', 'BA_x1', 'BA_x2', 'BA_x3', 'A', 'B'] (include second-order)

        Returns:
            labels_sample_matrices (list): labels of sample matrices
        """
        labels_sample_matrices = [
            'AB_' + self.parameter_names[i] for i in range(self.number_parameters)
        ]
        if self.calculate_second_order:
            sample_matrices_second_order = [
                'BA_' + self.parameter_names[i] for i in range(self.number_parameters)
            ]
            labels_sample_matrices = labels_sample_matrices + sample_matrices_second_order
        labels_sample_matrices.extend(['A', 'B'])

        # check size
        if self.calculate_second_order:
            number_of_sample_matrices = 2 * self.number_parameters + 2
        else:
            number_of_sample_matrices = self.number_parameters + 2

        if number_of_sample_matrices != len(labels_sample_matrices):
            raise AssertionError("Wrong size of sampling matrices.")

        return labels_sample_matrices

    def _draw_base_samples(self):
        """Draw base sample matrices A and B.

        Returns:
            A (ndarray): Saltelli A sample matrix
            B (ndarray): Saltelli B sample matrix
        """
        draw_sample = self._get_base_sample_method()
        base_sample_A, base_sample_B = draw_sample()

        A = self.parameters.inverse_cdf_transform(base_sample_A)
        B = self.parameters.inverse_cdf_transform(base_sample_B)

        return A, B

    def _get_base_sample_method(self):
        """Get sampling approach for base samples.

        Returns:
             function object (obj): function object for base sample type
        """
        if self.sampling_approach == "pseudo_random":
            return self._base_sample_pseudo_random
        if self.sampling_approach == "quasi_random":
            return self._base_sample_quasi_random

        raise ValueError(
            "Unknown sampling approach. Valid approaches are pseudo_random or quasi_random."
        )

    def _base_sample_quasi_random(self):
        """Sample with Sobol sequence.

        Get pseudo-random base samples based on Sobol sequence.

        For details on the Sobol sequence
        Owen, Art B. ‘On Dropping the First Sobol’ Point’. ArXiv:2008.08051 [Cs, Math, Stat], 2021.
        https://arxiv.org/abs/2008.08051v4.

        Returns:
            base_sample_A (ndarray): Saltelli A sample matrix in unit range
            base_sample_B (ndarray): Saltelli B sample matrix in unit range
        """
        sobol_engine = qmc.Sobol(
            d=2 * self.number_parameters, scramble=True, seed=self.seed_monte_carlo
        )
        base_sequence = sobol_engine.random(n=self.number_monte_carlo_samples)

        base_sample_A = base_sequence[:, : self.number_parameters]
        base_sample_B = base_sequence[:, self.number_parameters :]

        return base_sample_A, base_sample_B

    def _base_sample_pseudo_random(self):
        """Get pseudo-random base samples.

        Based on numpy random-number generator.

        Returns:
            base_sample_A (ndarray): Saltelli A sample matrix in unit range
            base_sample_B (ndarray): Saltelli B sample matrix in unit range
        """
        rng = np.random.default_rng(seed=self.seed_monte_carlo)
        base_sample_A = rng.uniform(0, 1, (self.number_monte_carlo_samples, self.number_parameters))
        base_sample_B = rng.uniform(0, 1, (self.number_monte_carlo_samples, self.number_parameters))

        return base_sample_A, base_sample_B


class ThirdOrderSampler(Sampler):
    """ThirdOrderSampler class.

    Draw Monte-Carlo samples and generate sample matrices following

    Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
    Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270.
    https://doi.org/10.1016/j.cpc.2009.09.018.

    Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
    John Wiley & Sons, Ltd., 2008.
    https://doi.org/10.1002/9780470725184.

    Attributes:
        calculate_third_order (bool): true if third-order indices only are calculated
        third_order_parameters (list): list of parameter combination for third-order index
    """

    def __init__(
        self,
        parameter_names,
        number_monte_carlo_samples,
        seed_monte_carlo,
        calculate_second_order,
        parameters,
        sampling_approach,
        calculate_third_order,
        third_order_parameters,
    ):
        """Initialize.

        Args:
            parameter_names (list): list of names of input parameters
            number_monte_carlo_samples (int): number of Monte-Carlo samples
            seed_monte_carlo (int): seed for random samples
            calculate_second_order (bool): true if second-order indices are calculated
            parameters (list): information about distribution of random variables
            sampling_approach (string): sampling approach (pseudo-random or quasi-random)
            calculate_third_order (bool): true if third-order indices only are calculated
            third_order_parameters (list): list of parameter combination for third-order index
        """
        super().__init__(
            parameter_names=parameter_names,
            number_monte_carlo_samples=number_monte_carlo_samples,
            seed_monte_carlo=seed_monte_carlo,
            calculate_second_order=calculate_second_order,
            parameters=parameters,
            sampling_approach=sampling_approach,
        )
        self.calculate_third_order = calculate_third_order
        self.third_order_parameters = third_order_parameters

    @classmethod
    def from_config_create(cls, method_options, parameter_names, parameters):
        """Create third-order sampler from problem description.

        Args:
            method_options (dict): dictionary with method options
            parameter_names (list): list of parameter names
            parameters (list): information about distribution of random variables

        Returns:
            sampler: Sampler object
        """
        number_monte_carlo_samples = method_options['number_monte_carlo_samples']
        _logger.info('Number of Monte-Carlo samples = %i', number_monte_carlo_samples)

        seed_monte_carlo = method_options.get('seed_monte_carlo', 42)

        calculate_second_order = method_options.get("second_order", False)
        sampling_approach = method_options.get('sampling_approach', 'quasi_random')

        calculate_third_order = method_options.get("third_order", False)
        third_order_parameters = method_options.get("third_order_parameters", None)

        return cls(
            parameter_names=parameter_names,
            number_monte_carlo_samples=number_monte_carlo_samples,
            seed_monte_carlo=seed_monte_carlo,
            calculate_second_order=calculate_second_order,
            parameters=parameters,
            sampling_approach=sampling_approach,
            calculate_third_order=calculate_third_order,
            third_order_parameters=third_order_parameters,
        )

    def sample(self):
        """Generate sample matrices.

        Generate sample matrices A, B, AB, BA, AB_ijk following

        Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
        Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
        (1 February 2010): 259–270. https://doi.org/10.1016/j.cpc.2009.09.018.

        Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
        John Wiley & Sons, Ltd., 2008. https://doi.org/10.1002/9780470725184.

        A:  sample matrix A
        B:  sample matrix B
        AB:  separated from A by only change in X_i; from B by change in X_{~i}
        BA:  separated from B by only change in X_i; from A by change in X_{~i}
        AB_ijk: separated from A by a change in X_i, X_j and X_k

        Returns:
            samples (xr.Array): Monte-Carlo sample matrices
        """
        samples = self._init_samples()

        third_order_parameter_indices = [
            self.parameter_names.index(parameter) for parameter in self.third_order_parameters
        ]

        A, B = self._draw_base_samples()
        samples.loc[{"sample_matrix": 'A'}] = A
        samples.loc[{"sample_matrix": 'B'}] = B

        for i, parameter_name in enumerate(self.third_order_parameters):
            AB = A.copy()
            AB[:, third_order_parameter_indices[i]] = B[
                :, third_order_parameter_indices[i]
            ].flatten()
            samples.loc[{"sample_matrix": 'AB_' + parameter_name}] = AB

            BA = B.copy()
            BA[:, third_order_parameter_indices[i]] = A[
                :, third_order_parameter_indices[i]
            ].flatten()
            samples.loc[{"sample_matrix": 'BA_' + parameter_name}] = BA

        # all columns from A except columns (i,j,k) for the third-order index interaction from B
        AB_ijk = A.copy()
        AB_ijk[:, third_order_parameter_indices] = B[:, third_order_parameter_indices]

        samples.loc[
            {"sample_matrix": ['AB_' + '_'.join(self.third_order_parameters)]}
        ] = np.expand_dims(AB_ijk, axis=1)

        return samples

    def _setup_sample_matrices_labels(self):
        """Set up labels of sample matrices.

        For further information on the sample matrices see

        Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
        Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
        (1 February 2010): 259–270.
        https://doi.org/10.1016/j.cpc.2009.09.018.

        Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
        John Wiley & Sons, Ltd., 2008.
        https://doi.org/10.1002/9780470725184.

        Example:
            ['AB_x1', 'AB_x2', 'AB_x3', 'BA_x1', 'BA_x2', 'BA_x3', 'AB_x1x2x3', 'A', 'B']

        Returns:
            labels_sample_matrices (list): labels of sample matrices
        """
        labels_sample_matrices = [
            'AB_' + parameter_name for parameter_name in self.third_order_parameters
        ]
        labels_sample_matrices.extend(
            ['BA_' + parameter_name for parameter_name in self.third_order_parameters]
        )
        labels_sample_matrices.extend(['AB_' + '_'.join(self.third_order_parameters)])
        labels_sample_matrices.extend(['A', 'B'])
        return labels_sample_matrices
