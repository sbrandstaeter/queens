import copy
import logging

import matplotlib.pyplot as plt
import numpy as np

import pqueens.database.database as DB_module
from pqueens.external_geometry import from_config_create_external_geometry
from pqueens.models import from_config_create_model
from pqueens.randomfields.univariate_field_generator_factory import (
    UniVarRandomFieldGeneratorFactory,
)
from pqueens.utils import mcmc_utils
from pqueens.utils.get_random_variables import get_random_samples
from pqueens.utils.process_outputs import process_ouputs, write_results

from .iterator import Iterator

_logger = logging.getLogger(__name__)


class MonteCarloIterator(Iterator):
    """Basic Monte Carlo Iterator to enable MC sampling.

    Attributes:
        model (model):              Model to be evaluated by iterator
        seed  (int):                Seed for random number generation
        num_samples (int):          Number of samples to compute
        result_description (dict):  Description of desired results
        samples (np.array):         Array with all samples
        outputs (np.array):         Array with all model outputs
        external_geometry_obj (obj): External external_geometry_obj object containing node and
                                     mesh information
        db (obj):                   Data base object
    """

    def __init__(
        self,
        model,
        seed,
        num_samples,
        result_description,
        global_settings,
        external_geometry_obj,
        db,
    ):
        super(MonteCarloIterator, self).__init__(model, global_settings)
        self.seed = seed
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.external_geometry_obj = external_geometry_obj
        self.db = db

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """Create MC iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: MonteCarloIterator object
        """
        print(config.get('experiment_name'))
        if iterator_name is None:
            method_options = config['method']['method_options']
        else:
            method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)
        if config.get('external_geometry') is not None:
            external_geometry_obj = from_config_create_external_geometry(config)
        else:
            external_geometry_obj = None

        db = DB_module.database

        return cls(
            model,
            method_options['seed'],
            method_options['num_samples'],
            result_description,
            global_settings,
            external_geometry_obj,
            db,
        )

    def eval_model(self):
        """Evaluate the model."""
        return self.model.evaluate()

    def pre_run(self):
        """Generate samples for subsequent MC analysis and update model."""
        np.random.seed(self.seed)

        parameters = self.model.get_parameter()
        num_inputs = 0
        # get random Variables
        random_variables = parameters.get('random_variables', None)
        # get number of rv
        if random_variables is not None:
            num_rv = len(random_variables)
        num_inputs += num_rv

        # get random fields
        random_fields = parameters.get('random_fields', None)

        num_eval_locations = []

        # TODO this if statement is useless for external random fields
        if random_fields is not None:
            for _, rf in random_fields.items():
                # TODO not nice and should be changed
                external_bool = rf.get('external_definition', False)
                if not external_bool:
                    dim = rf["dimension"]
                    eval_locations_list = rf.get("eval_locations", None)  # TODO how to handle this?
                    eval_locations = np.array(eval_locations_list).reshape(-1, dim)
                    temp = eval_locations.shape[0]
                    num_eval_locations.append(temp)
                    num_inputs += temp

        self.samples = np.zeros((self.num_samples, num_inputs))
        # loop over random variables to generate samples
        i = 0
        for _, rv in random_variables.items():
            rv_size = rv['size']
            if rv_size != 1:
                raise RuntimeError("Multidimensional random variables are not supported yet.")
            # TODO once the above restriction is loosened take care of the indexing!
            #  and the squeeze
            self.samples[:, i] = np.squeeze(get_random_samples(rv, self.num_samples))
            i += 1

        # loop over random fields to generate samples
        field_num = 0
        if random_fields is not None:
            for rf_name, rf in random_fields.items():
                print("The selected random field is: {}".format(rf.get("corrstruct")))
                # create appropriate random field generator

                random_field_opt = {}
                random_field_opt['corrstruct'] = rf.get("corrstruct")
                random_field_opt['corr_length'] = rf.get("corr_length")
                # TODO logic is ugly and should be changed in the future
                if rf.get("corrstruct") == "generic_external_random_field":
                    random_field_opt['std_hyperparam_rf'] = rf['std_hyperparam_rf']
                    random_field_opt['mean_fun_type'] = rf['mean_fun_type']
                    random_field_opt['mean_fun_params'] = rf['mean_fun_params']
                    random_field_opt['num_samples'] = self.num_samples
                    random_field_opt['dimension'] = rf.get("dimension")
                    # get input points form external external_geometry_obj here
                    if self.external_geometry_obj is not None:
                        self.external_geometry_obj.main_run()
                    random_field_opt['external_geometry_obj'] = self.external_geometry_obj
                    random_field_opt['external_definition'] = rf['external_definition']
                else:
                    random_field_opt['eval_locations'] = rf.get('eval_location')  # TODO depreciated
                    random_field_opt['dimension'] = rf.get("dimension")
                    random_field_opt['energy_frac'] = rf.get("energy_frac")
                    random_field_opt['field_bbox'] = np.array(rf.get("field_bbox"))
                    random_field_opt['num_terms_per_dim'] = rf.get("num_terms_per_dim")
                    random_field_opt['total_terms'] = rf.get("total_terms")
                # pylint: disable=line-too-long
                my_field_generator = (
                    UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
                        mcmc_utils.create_proposal_distribution(rf), **random_field_opt
                    )
                )
                # pylint: enable=line-too-long
                # eval_locations_list = rf.get("eval_locations", None)
                # eval_locations = np.array(eval_locations_list).reshape(
                #    -1, random_field_opt['dimension']
                # )

                if random_field_opt['corrstruct'] == 'generic_external_random_field':

                    my_field_generator.main_run()  # here the field truncation is done by main run
                    my_vals = np.atleast_2d(my_field_generator.realizations)
                    self.samples = np.hstack((self.samples, np.atleast_2d(my_vals).T))

                    # db code comes here
                    truncated_rf_dict = {
                        rf_name: [
                            my_field_generator.weighted_eigen_val_mat_truncated,
                            my_field_generator.mean,
                        ]
                    }
                    experiment_field = "truncated_random_fields"
                    batch = 1  # dummy batch
                    self.db.save(
                        truncated_rf_dict,
                        self.global_settings["experiment_name"],
                        experiment_field,
                        batch,
                    )

                    # initialize here sizes of all random field samples
                    self.model.variables[0].variables[rf_name].update(
                        {"size": my_field_generator.weighted_eigen_val_mat_truncated.shape[1]}
                    )
                    for i in range(1, self.num_samples):
                        self.model.variables.append(copy.deepcopy(self.model.variables[0]))

                else:
                    my_stoch_dim = my_field_generator.get_stoch_dim()
                    my_vals = np.zeros((self.num_samples, eval_locations.shape[0]))

                    for i in range(self.num_samples):
                        xi = np.random.randn(my_stoch_dim, 1)
                        my_vals[i, :] = my_field_generator.evaluate_field_at_location(
                            eval_locations, xi
                        )
                    self.samples[
                        :, num_rv + field_num : num_rv + field_num + len(eval_locations)
                    ] = my_vals

                field_num += 1

    def core_run(self):
        """Run Monte Carlo Analysis on model."""
        self.model.update_model_from_sample_batch(self.samples)
        self.output = self.eval_model()

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description, self.samples)
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

                # ------------------------------ WIP PLOT OPTIONS -----------------------------
                if self.result_description['plot_results'] is True:
                    # Check for dimensionality of the results
                    plt.rcParams["mathtext.fontset"] = "cm"
                    plt.rcParams.update({'font.size': 23})
                    fig, ax = plt.subplots()

                    if results['raw_output_data']['mean'][0].shape[0] > 1:
                        for ele in results['raw_output_data']['mean']:
                            ax.plot(ele[:, 0], ele[:, 1])

                        ax.set_xlabel(r't [s]')
                        ax.set_ylabel(r'$C_L(t)$')
                        plt.show()
                    else:
                        data = results['raw_output_data']['mean']
                        ax.hist(data, bins=200)
                        ax.set_xlabel(r'Count [-]')
                        ax.set_xlabel(r'$C_L(t)$')
                        plt.show()
        # else:
        _logger.debug("Size of inputs {}".format(self.samples.shape))
        _logger.debug("Inputs {}".format(self.samples))
        _logger.debug("Size of outputs {}".format(self.output['mean'].shape))
        _logger.debug("Outputs {}".format(self.output['mean']))
