import numpy as np
from pqueens.models.bmfmc_model import BMFMCModel
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from .iterator import Iterator
from pqueens.utils.process_outputs import process_ouputs
from pqueens.utils.process_outputs import write_results
from .scale_samples import scale_samples
from pqueens.database import mongodb as db
from pqueens.iterators.data_iterator import DataIterator
import os

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
    def __init__(self, lf_data_iterators, hf_data_iterators, result_description, experiment_dir):

        super(BmfmcIterator, self).__init__() # Input prescribed by iterator.py
        self.model = None
        self.result_description = result_description
        self.lf_data_iterators = lf_data_iterators
        self.hf_data_iterator = hf_data_iterator
        self.experiment_dir = experiment_dir
        self.train_in = None
        self.hf_train_out = None
        self.lfs_train_out = None
        self.lf_mc_in = None
        self.lfs_mc_out = None
        self.output = None # this is still important and will prob be the marginal pdf of the hf / its statistics

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None): #TODO: the model arg here seems an old relict. we bypass it here
        """ Create LHS iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section in options dict
            model (model): model to use

        Returns:
            iterator: BmfmcIterator object

        """
        # Initialize Iterator and model
        if iterator_name is None:
            method_options = config["method"]["method_options"]
            print("Method options {}".format(method_options))
        else:
            method_options = config[iterator_name]["method_options"]
        result_description = method_options.get("result_description", None)
        experiment_dir = config["method"]["method_options"]["experiment_dir"]

        # Create the data iterator from config file
        lf_data_paths = method_options["path_to_lf_data"] # necessary to substract that name from name list of all models for evaluation

        hf_data_path = method_options["path_to_hf_data"]

        lf_data_iterators = DataIterator(lf_data_paths, None, None) #TODO modify data iterator so that a list of iterator objects can be returned if a list of paths is given
        hf_data_iterator = DataIterator(hf_data_path, None, None)

        return cls(lf_data_iterators, hf_data_iterator, result_description, experiment_dir)

    def core_run(self):
        """ Generate simulation runs for lofis and hifis that cover the output
            space based on one p(y) distribution of one lofi """
        # extract one input matrix(each input is a vector and we collect several inputs)
        self.lf_mc_in = self.lf_data_iterators[0].read_pickle_file()[0] # here we assume that data is cleaned up and all lfs have the same inputs x
        # collect the output values of the LF MC data into one array
        self.lfs_mc_out = [lf_data_iterator.read_pickle_file()[1] for lf_data_iterator,_ in enumerate(self.lf_data_iterators)] # list of lf data with another list or dictionary per lf containing x_vec and y(_vec)

####### HERE create the HF initial design runs if not already done ##############################
        if self.hf_data_iterator == None:
#################################################
###### TODO: This needs still to be done!!!!!!!##
#################################################
            minval = mc_y.min()
            maxval = mc_y.max()
            n_bins = self.num_samples//2
            break_points = np.linspace(minval, maxval, n_bins+1)

            # Binning using cut function of pandas -> gives every y a bin number
            bin_vec = pd.cut(mc_y, bins=break_points, labels=False,
                            include_lowest=True, retbins=True)

           # calculate most distant point pair per bin
            mapping_points_x = []
            mapping_points_y = []
            for bin_n in range(self.num_samples//2):
                # create boolean vector with length of y that filters all y beloning to one active bin
                boolean_vec = [bin_vec==bin_n] # Check if this is enough to create a boolean vec or if for loop is necessary
                bin_data_x = mc_x[boolean_vec]
                bin_data_y = mc_y[boolean_vec]

               # (furtherst neighbor algorithm), design of data: y,x1,x2,x3....
                D = pdist(bin_data_x)
                D = squareform(D)
                [row, col] = np.unravel_index(np.argmax(D), D.shape)
                mapping_points_x.append(bin_data_x[row, col])
                mapping_points_y.append(bin_data_y[row, col])

           # self.samples = mapping_points_x
            self.mc_map_output = mapping_points_y
######### HERE just load HF data if already there, find the LF data that belongs to it and write a new pickle file
        else:
           # check if training data file already exists if not write it
            path_to_train_data = os.path.join(self.experiment_dir,'hf_lf_train.pickle')
            exists = os.path.isfile(path_to_train_data)
            if exists:
                with open(path_to_train_data,'rb') as handle: #TODO: give better name according to experiment and choose better location
                    self.train_in, self.lfs_train_out, self.hf_train_out, self.lf_mc_in, self.lfs_mc_out = pickle.load(handle) #TODO --> Change this according to design below!!
            else:
                self.train_in, self.hf_train_out = self.hf_data_iterator.read_pickle()
                # find the touples of lf and hf data that match
                bolean_vec = np.isin(lf_mc_in, self.train_in) # TODO: probably we need to change the shapes
                self.lfs_train_out = self.lfs_mc_out[:][boolean_vec, :] #TODO Check if the list works out here with : or if we need to iterate

                # write new pickle file for subsequent analysis with matching data
                with open(path_to_train_data,'wb') as handle: #TODO: give better name according to experiment and choose better location
                pickle.dump([self.train_in, self.lfs_train_out, self.hf_train_out, self.lf_mc_in, self.lfs_mc_out],handle,protocol=pickle.HIGHEST_PROTOCOL) #TODO: check if list is right format here
        # CREATE the underlying model
         bmfmc_model = BMFMCModel.from_config_create_model(config, self.train_in, self.lfs_train_out, self.hf_train_out, self.lf_mc_in, self.lfs_mc_out) # TODO: HERE we actualluy define the BMFMC_model and bypass the input arg

####### TODO: This needs still to be implemented to run HF directly from this iterator for initial desing
        """ Run Analysis on all models """
        # here the set of new input variables will be passed to all lofis/hifi--> previous list_iterator not necessary!
#        self.model.update_model_from_sample_batch(self.samples)

        # here all models will be evaluated (multifidelity  model will be evaluated)
        # output matrix has the form:[ ylofi1, ylofi2, ..., yhifi ]
 #       self.output = self.eval_model
         # What about the already calculated points?
         #TODO some active learning here


    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()


    def active_learning(self):
        pass

    def post_run(self):
        """ Analyze the results """
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description)
            write_results(results,
                              self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])
        #else:
        #print("Size of inputs {}".format(self.samples.shape))
        #print("Inputs {}".format(self.samples))
        print("Size of outputs {}".format(self.output['mean'].shape))
        print("Outputs {}".format(self.output['mean']))
