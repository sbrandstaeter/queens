import numpy as np
from .data_iterator import DataIterator
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from pyDOE import lhs
from .iterator import Iterator
from pqueens.models.model import Model # in this case we have the BMFMC model
from pqueens.utils.process_outputs import process_ouputs
from pqueens.utils.process_outputs import write_results
from .scale_samples import scale_samples
from pqueens.database import mongodb as db
class BmfmcIterator(Iterator):
    """ Basic BMFMC Iterator to enable selective sampling for the hifi-lofi
        model (the basis was a standard LHCIterator). Additionally the BMFMC
        Iterator contains the subiterator data_iterator to make use of already
        performed MC samples on the lo-fi models

    Attributes:
        model (model):        Model to be evaluated by iterator
        seed  (int):          Seed for random number generation
        num_samples (int):    Number of samples to compute
        num_iterations (int): Number of optimization iterations of design
        result_description (dict):  Description of desired results
        samples (np.array):   Array with all samples
        outputs (np.array):   Array with all model outputs

    """
    def __init__(self, model, seed, num_samples, result_description, global_settings, data_iterator):
        """ Initialize the BMFMCIterator

        Args:
            model:
            seed:
            num_samples:
            result_description:
            global_settings:
            data_iterator (Iterator): Sub_iterator to enable reading data form pickle file
        """

        super(BmfmcIterator, self).__init__(model, global_settings)
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.data_iterator = data_iterator

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """ Create LHS iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: BmfmcIterator object

        """
        if iterator_name is None:
            method_options = config["method"]["method_options"]
        else:
            method_options = config[iterator_name]["method_options"]
        if model is None:
            model_name = method_options["model"]
            model = Model.from_config_create_model(model_name, config)

        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)

        data_iterator_name = method_options["data_iterator"]
        data_model = method_options["data_model"]
        # create data_iterator
        data_iterator = DataIterator.from_config_create_iterator(config,data_iterator_name,data_model) #TODO What is data_model supposed to be?

        return cls(model, method_options["num_samples"],
                   result_description,
                   global_settings, data_iterator)

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def binning(self,data, num_samples, column):
        # data is assumed to be a np array
        # Define min and max values:
        minval = data[:,column].min()
        maxval = data[:,column].max()
        n_bins = num_samples//2
        break_points = np.linspace(minval,maxval,n_bins+1)

       # if no labels provided, use default labels 0 ... (n-1)
       # if not labels:
       #    labels = range(n_bins+1)

       # Binning using cut function of pandas
        colBin = pd.cut(data, bins=break_points, labels=False,
                        include_lowest=True, retbins=True)
        return colBin

    def furthest_points(self, Xdata):
        D = pdist(Xdata)
        D = squareform(D)
        [row,col] = np.unravel_index(np.argmax(D), D.shape)
        return [row, col]

    def pre_run(self):
        """ Generate simulation runs for lofis and hifis that cover the output
        space based on one p(y) distribution of one lofi """

        # load data for lofi on which was sampled from db or pickle file
        # The following data structure is assumed:
            # y x1 x2 x3 ....xn
        [samples, output] = self.data_iterator.read_pickle_file # TODO: Check how to load pickle file
        data_lofi_mc = np.array(output, samples)

        # Divide that range into 10 bins TODO:change that to input file at some
        # point
        binned_data_lofi_mc = self.binning(data_lofi_mc,self.num_samples,1)

        # Within each bin select m y_i whose X are furthest apart in the input
        mapping_points=[]
        for bin_n in range(20):
            bin_data = data_lofi_mc[binned_data_lofi_mc==bin_n]
            # (furtherst neighbor algorithm), design of data: y,x1,x2,x3....
            row, col = self.furthest_points(bin_data)
            mapping_points.append(bin_data[row, :], axis=0)
            mapping_points.append(bin_data[col, :], axis=0)
            # desing of maping points: x1,x2,x3,x4... per line

        self.samples = mapping_points
######### Here evaluation of hifi and lofis #####################
    def core_run(self):
        """ Run Analysis on all models """
        # TODO: The model function in BMFMC model should direclty initiate a
        # evaluation an all models and later return the output of all models

        # here input variables will be passed to all lofis/hifi

        self.model.update_model_from_sample_batch(self.samples)

        # here all models will be evaluated (BMFMC model will be evaluated)
        # output matrix has the form:[ ylofi1, ylofi2, ..., yhifi ]
        self.output = self.eval_model

######### save the evaluation in distinct pickle file ############

    def post_run(self):
        """ Analyze the results """
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description)
            if self.result_description["write_results"] is True:
                write_results(results,
                              self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])
        #else:
        print("Size of inputs {}".format(self.samples.shape))
        print("Inputs {}".format(self.samples))
        print("Size of outputs {}".format(self.output['mean'].shape))
        print("Outputs {}".format(self.output['mean']))
