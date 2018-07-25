
import random
from scipy.stats import norm
import numpy as np
from pqueens.models.model import Model
from .iterator import Iterator
from . import sobol_sequence
from .scale_samples import scale_samples
from pqueens.utils.process_outputs import write_results


class SaltelliIterator(Iterator):
    """ Pseudo Saltelli iterator

        This class is for performing sensitivity analysis using Sobol indices
        based on a Saltelli sampling scheme.

    References:

    [1] Sobol, I. M. (2001).  "Global sensitivity indices for nonlinear
        mathematical models and their Monte Carlo estimates."  Mathematics
        and Computers in Simulation, 55(1-3):271-280,
        doi:10.1016/S0378-4754(00)00270-6.

    [2] Saltelli, A. (2002).  "Making best use of model evaluations to
        compute sensitivity indices."  Computer Physics Communications,
        145(2):280-297, doi:10.1016/S0010-4655(02)00280-1.

    [3] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
        S. Tarantola (2010).  "Variance based sensitivity analysis of model
        output.  Design and estimator for the total sensitivity index."
        Computer Physics Communications, 181(2):259-270,
        doi:10.1016/j.cpc.2009.09.018.

    Attributes:
        seed (int):                     Seed for random number generator
        num_samples (int):              Number of samples
        calc_second_order (bool):       Calculate second-order sensitivities
        num_bootstrap_samples (int):    Number of bootstrap samples
        confidence_level (float):       The confidence interval level
        samples (np.array):             Array with all samples
        output (dict):                  Dict with all model outputs
        num_params (int):               Number of uncertain model parameters
        parameter_names (list):         List of parameter names
        sensitivity_incides (dict):     Dictionary holdin sensitivity incides
    """

    def __init__(self, model, seed, num_samples, calc_second_order,
                 num_bootstrap_samples, confidence_level, result_description,
                 global_settings):
        """ Initialize Saltelli iterator object

        Args:
            model (model):                  Model to sample
            seed (int):                     Seed for random number generation
            num_samples (int):              Number of desired (random) samples
            calc_second_order (bool):       Calculate second-order sensitivities
            num_bootstrap_samples (int):    Number of bootstrap samples
            result_description (dict):      Description of desired results
            confidence_level (float):       The confidence interval level
        """
        super(SaltelliIterator, self).__init__(model, global_settings)

        self.num_samples = num_samples
        self.seed = seed
        self.calc_second_order = calc_second_order
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.num_params = None
        self.parameter_names = []

    @classmethod
    def from_config_create_iterator(cls, config, model=None):
        """ Create Saltelli iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: SaltelliIterator object

        """
        method_options = config["method"]["method_options"]

        if model is None:
            model_name = method_options["model"]
            model = Model.from_config_create_model(model_name, config)

        return cls(model, method_options["seed"],
                   method_options["num_samples"],
                   method_options["calc_second_order"],
                   method_options["num_bootstrap_samples"],
                   method_options["confidence_level"],
                   method_options.get("result_description", None),
                   config["global_settings"])

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generates samples based on Saltelli's extension of the Sobol sequence

            Saltelli's scheme extends the Sobol sequence in a way to reduce
            the error rates in the resulting sensitivity index calculations. If
            calc_second_order is False, the resulting matrix has N * (D + 2)
            rows, where D is the number of parameters and N is the number of
            samples. If calc_second_order is True, the resulting matrix has
            N * (2D + 2) rows.

        """
        # fix seed of random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)

        parameters = self.model.get_parameter()
        random_variables = parameters.get("random_variables", None)
        # get number of rv
        if random_variables is not None:
            num_rv = len(random_variables)
        self.num_params = num_rv
        # get random fields
        random_fields = parameters.get("random_fields", None)
        if random_fields is not None:
            raise RuntimeError("The SaltelliIterator does not work in conjunction with random fields.")

        # loop over random variables to generate samples
        distribution_info = []
        for name, rv in random_variables.items():
            # store name
            self.parameter_names.append(name)
            # get appropriate random number generator
            temp = {}
            temp["distribution"] = rv["distribution"]
            temp["distribution_parameter"] = rv["distribution_parameter"]
            distribution_info.append(temp)


        self.sensitivity_incides = self.create_si_dict()

        # How many values of the Sobol sequence to skip
        skip_values = 1000

        # Create base sequence - could in principle be any type of sampling
        # TODO add lhs sampling to this
        base_sequence = sobol_sequence.sample(self.num_samples + skip_values, 2 * self.num_params)

        if self.calc_second_order:
            saltelli_sequence = np.zeros([(2 * self.num_params + 2) * \
                self.num_samples, self.num_params])
        else:
            saltelli_sequence = np.zeros([(self.num_params + 2) * \
                self.num_samples, self.num_params])

        # The samples in saltelli sequence are the matrices A and B in
        # Saltelli et al. 2010. We now have to compute  the matrices A_B^i and
        # B_A^i, see Saltelli et al. section 3. This is done in the following.
        index = 0
        for i in range(skip_values, self.num_samples + skip_values):
            # Copy matrix "A"
            for j in range(self.num_params):
                saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

            # Cross-sample elements of "B" into "A"
            for k in range(self.num_params):
                for j in range(self.num_params):
                    if  j == k:
                        saltelli_sequence[index, j] = base_sequence[i, j + self.num_params]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j]

                index += 1

            # Cross-sample elements of "A" into "B"
            # Only needed if you're doing second-order indices (true by default)
            if self.calc_second_order:
                for k in range(self.num_params):
                    for j in range(self.num_params):
                        if j == k:
                            saltelli_sequence[index, j] = base_sequence[i, j]
                        else:
                            saltelli_sequence[index, j] = base_sequence[i, j + self.num_params]
                    index += 1

            # Copy matrix "B"
            for j in range(self.num_params):
                saltelli_sequence[index, j] = base_sequence[i, j + self.num_params]

            index += 1


        # scaling values to other distributions
        scaled_saltelli = scale_samples(saltelli_sequence, distribution_info)
        self.samples = scaled_saltelli

    def get_all_samples(self):
        """ Return all samples

        Returns:
            np.array:    array with all samples

        """
        return self.samples

    def core_run(self):
        """ Run Analysis on model """

        # update the model
        self.model.update_model_from_sample_batch(self.samples)

        # evaluate
        self.output = self.eval_model()

        # analyse
        self.analyze(np.reshape(self.output['mean'],(-1)))

    def post_run(self):
        """ Analyze the results """
        results = self.process_results()
        if self.result_description is not None:
            if self.result_description["write_results"] is True:
                write_results(results, self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])
            else:
                self.print_results(results)

    def analyze(self, Y):
        """ Perform Sobol Analysis on model outputs.

        Computes a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf', where
        each entry is a list of size num_params containing the
        indices in the same order as the parameter file.  If calc_second_order is
        True, the dictionary also contains keys 'S2' and 'S2_conf'.

        """

        if self.calc_second_order and Y.size % (2 * self.num_params + 2) == 0:
            N = int(Y.size / (2 * self.num_params + 2))
        elif not self.calc_second_order and Y.size % (self.num_params + 2) == 0:
            N = int(Y.size / (self.num_params + 2))
        else:
            raise RuntimeError(""" Incorrect number of samples.""")

        if self.confidence_level < 0 or self.confidence_level > 1:
            raise RuntimeError("Confidence level must be between 0-1.")

        # normalize the model outputs
        Y = (Y - Y.mean())/Y.std()
        A, B, AB, BA = self.separate_output_values(Y)

        r = np.random.randint(N, size=(N, self.num_bootstrap_samples))
        Z = norm.ppf(0.5 + self.confidence_level / 2)

        S = self.sensitivity_incides

        for j in range(self.num_params):
            S['S1'][j] = self.first_order(A, AB[:, j], B)
            S['S1_conf'][j] = Z * self.first_order(A[r], AB[r, j], B[r]).std(ddof=1)
            S['ST'][j] = self.total_order(A, AB[:, j], B)
            S['ST_conf'][j] = Z * self.total_order(A[r], AB[r, j], B[r]).std(ddof=1)

        # Second order (+conf.)
        if self.calc_second_order:
            for j in range(self.num_params):
                for k in range(j + 1, self.num_params):
                    S['S2'][j, k] = self.second_order(
                        A, AB[:, j], AB[:, k], BA[:, j], B)
                    S['S2_conf'][j, k] = Z * self.second_order(A[r], AB[r, j],
                        AB[r, k], BA[r, j], B[r]).std(ddof=1)

        self.sensitivity_incides = S


    def first_order(self, A, AB, B):
        """ Compute first order indices, normalized by sample variance

        Args:
            A (np.array):   Results corresponding to A
            AB (np.array):  Results corresponding to AB
            B (np.array):   Results corresponding B

        Returns:
            np.array: first order indices
        """
        return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)


    def total_order(self, A, AB, B):
        """ Compute total order indices, normalized by sample variance

        Args:
            A (np.array):   Results corresponding to A
            AB (np.array):  Results corresponding to AB
            B (np.array):   Results corresponding B

        Returns:
            np.array: total order indices

        """
        return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)


    def second_order(self, A, ABj, ABk, BAj, B):
        """ Compute second order indices, normalized by sample variance

        Args:
            A (np.array):   Results corresponding to A
            AB (np.array):  Results corresponding to AB
            B (np.array):   Results corresponding B

        Returns:
            np.array: Second order indices

        """
        Vjk = np.mean(BAj * ABk - A * B, axis=0) / np.var(np.r_[A, B], axis=0)
        Sj = self.first_order(A, ABj, B)
        Sk = self.first_order(A, ABk, B)

        return Vjk - Sj - Sk


    def create_si_dict(self):
        """ Create a dictionnary to store the results

        Returns:
            dict: Prototype dictionay to store sensitivity indices
        """
        S = dict((k, np.zeros(self.num_params)) for k in ('S1','S1_conf','ST','ST_conf'))

        if self.calc_second_order:
            S['S2'] = np.zeros((self.num_params, self.num_params))
            S['S2'][:] = np.nan
            S['S2_conf'] = np.zeros((self.num_params, self.num_params))
            S['S2_conf'][:] = np.nan
        return S

    def separate_output_values(self, Y):
        """ From all computed samples in Y get results corresponding to the
        matrices A, B, AB, BA, see Saltelli et a. 2010.

        Args:
            Y (np.array): Outputs from model
        Returns:

        """

        AB = np.zeros((self.num_samples, self.num_params))
        BA = np.zeros((self.num_samples, self.num_params)) if self.calc_second_order else None
        step = 2 * self.num_params + 2 if self.calc_second_order else self.num_params + 2

        A = Y[0:Y.size:step]
        B = Y[(step - 1):Y.size:step]
        for j in range(self.num_params):
            AB[:, j] = Y[(j + 1):Y.size:step]
            if self.calc_second_order:
                BA[:, j] = Y[(j + 1 + self.num_params):Y.size:step]

        return A, B, AB, BA

    def print_results(self, results):
        """ Function to print results """

        # get shorter name
        S = results["sensitivity_incides"]
        parameter_names = results["parameter_names"]
        title = 'Parameter'
        print('%s   S1       S1_conf    ST    ST_conf' % title)
        j = 0
        for name in parameter_names:
            print('%s %f %f %f %f' % (name + '       ', S['S1'][j], S['S1_conf'][j],
                                      S['ST'][j], S['ST_conf'][j]))
            j = j+1

        if self.calc_second_order:
            print('\n%s_1 %s_2    S2      S2_conf' % (title, title))
            for j in range(self.num_params):
                for k in range(j + 1, self.num_params):
                    print("%s %s %f %f" % (parameter_names[j] + '            ', parameter_names[k] + '      ',
                                           S['S2'][j, k], S['S2_conf'][j, k]))


    def process_results(self):
        """ Write all results to self contained dictionary """

        results = {}
        results["parameter_names"] = self.parameter_names
        results["sensitivity_incides"] = self.sensitivity_incides
        results["second_order"] = self.calc_second_order

        return results
