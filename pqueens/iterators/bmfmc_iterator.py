import numpy as np
from pqueens.models.multifidelity_model import MultifidelityModel
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from pyDOE import lhs
from .iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.process_outputs import process_ouputs
from pqueens.utils.process_outputs import write_results
from .scale_samples import scale_samples
from pqueens.database import mongodb as db
from pqueens.iterators.data_iterator import DataIterator

class BmfmcIterator(Iterator):
    """ Basic BMFMC Iterator to enable selective sampling for the hifi-lofi
        model (the basis was a standard LHCIterator). Additionally the BMFMC
        Iterator contains the subiterator data_iterator to make use of already
        performed MC samples on the lo-fi models

    Attributes:
        model (model):        multi-fidelity model comprising sub-models
        num_samples (int):    Number of samples to compute
        path_to_lofi_mc_data:
        name_of_mc_model:
        path_to_point_list:
        result_description (dict):  Description of desired results (write data)
        samples (np.array):   Array with all samples
        outputs (np.array):   Array with all model outputs

    """
    def __init__(self, model, num_samples, mc_data_iterator, result_description, global_settings):

        super(BmfmcIterator, self).__init__(model, global_settings) # Input prescribed by iterator.py
        if type(self.model) is not MultifidelityModel:
            raise RuntimeError("BMFMCIterator requires a multi-fidelity model")

        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.mc_map_output = None
        self.output = None
        self.data_iterator = mc_data_iterator #  data iterator that hold all models mc eas run on

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """ Create LHS iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section in options dict
            model (model): model to use

        Returns:
            iterator: BmfmcIterator object

        """
        # Initialize Iterator options of "method group"
        if iterator_name is None:
            method_options = config["method"]["method_options"]
            print("Method options {}".format(method_options))

        else:
            method_options = config[iterator_name]["method_options"]
        if model is None:
            model_name = method_options["model"]
            model = Model.from_config_create_model(model_name, config)

        result_description = method_options.get("result_description", None)

        # create sub_iterators being either data_iterator or point_list_iterator types
        ## new stuff

        # Create the data iterator from config file
        mc_model_name = method_options["name_of_mc_model"] # necessary to substract that name from name list of all models for evaluation
        mc_data_path = method_options["path_to_mc_data"]
        data_iterator = DataIterator(mc_data_path, None, None) # None: result_description, global settings

#        # create the point list iterator with rest of models and point list location
#        all_model_names = config[model_name]["model_hierarchy"]
#        active_model_names = all_model_names.remove(mc_model_name)
#        path_to_point_list =  method_options["path_to_point_list"]
#        point_list_iterators = []
#        for name in active_model_names:
#            point_list_iterators.append(PointListIterator(name, path_to_point_list)) #TODO: Check if input for init correct
#
        return cls(model, method_options["num_samples"], data_iterator, result_description, config["global_settings"])

    def eval_model(self):
        """ Evaluate the model """
        evaluation = self.model.evaluate()
        mc_lofi_evaluation = [self.samples, self.mc_map_output] # TODO: Check if eval is really returne as a list of two arrays
        evaluation = [mc_lofi_evaluation, evaluation]
        return evaluation

    def pre_run(self):
        """ Generate simulation runs for lofis and hifis that cover the output
            space based on one p(y) distribution of one lofi """

        # data is assumed to be a np array
        # Define min and max values:
        mc_x, mc_y = self.data_iterator.read_pickle_file()
        minval = mc_y.min()
        maxval = mc_y.max()
        n_bins = self.num_samples//2
        break_points = np.linspace(minval, maxval, n_bins+1)

        # Binning using cut function of pandas
        bin_vec = pd.cut(mc_y, bins=break_points, labels=False,
                        include_lowest=True, retbins=True)

       # calculate most distant point pair per bin
        mapping_points_x = []
        mapping_points_y = []
        for bin_n in range(self.num_samples//2):
            boolean_vec = [bin_vec==bin_n] # Check if this is enough to create a boolean vec or if for loop is necessary
            bin_data_x = mc_x[boolean_vec]
            bin_data_y = mc_y[boolean_vec]

           # (furtherst neighbor algorithm), design of data: y,x1,x2,x3....
            D = pdist(bin_data_x)
            D = squareform(D)
            [row, col] = np.unravel_index(np.argmax(D), D.shape)
            mapping_points_x.append(bin_data_x[row, col])
            mapping_points_y.append(bin_data_y[row, col])

        self.samples = mapping_points_x
        self.mc_map_output = mapping_points_y

        return bin_vec # vector with bin numbers per y data point

######### Here evaluation of hifi and lofis #####################
    def core_run(self):
        """ Run Analysis on all models """
        # here the set of new input variables will be passed to all lofis/hifi--> previous list_iterator not necessary!
        self.model.update_model_from_sample_batch(self.samples)

        # here all models will be evaluated (multifidelity  model will be evaluated)
        # output matrix has the form:[ ylofi1, ylofi2, ..., yhifi ]
        self.output = self.eval_model
         # What about the already calculated points?
######### save the evaluation in distinct pickle file ############

    def post_run(self):
        """ Analyze the results """
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description)
            write_results(results,
                              self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])
        #else:
        print("Size of inputs {}".format(self.samples.shape))
        print("Inputs {}".format(self.samples))
        print("Size of outputs {}".format(self.output['mean'].shape))
        print("Outputs {}".format(self.output['mean']))
