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
import pandas as pd
from random import randint
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

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
    def __init__(self, config,lf_data_iterators, hf_data_iterator, result_description, experiment_dir, initial_design, predictive_var,BMFMC_reference,global_settings):

        super(BmfmcIterator, self).__init__(None,global_settings) # Input prescribed by iterator.py
        self.model = None
        self.result_description = result_description
        self.lf_data_iterators = lf_data_iterators
        self.hf_data_iterator = hf_data_iterator
        self.experiment_dir = experiment_dir
        self.train_in = None
        self.hf_train_out = None
        self.lfs_train_out = None
        self.lf_mc_in = None
        self.hf_mc = None
        self.lfs_mc_out = None
        self.output = None # this is still important and will prob be the marginal pdf of the hf / its statistics
        self.initial_design = initial_design
        self.predictive_var=predictive_var
        self.config = config # This is needed to construct the model on runtime
        self.BMFMC_reference = BMFMC_reference

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
        method_options = config["method"]["method_options"]
        BMFMC_reference = method_options["BMFMC_reference"]
        result_description = method_options["result_description"]
        experiment_dir = method_options["experiment_dir"]
        predictive_var = method_options["predictive_var"]

        # Create the data iterator from config file
        lf_data_paths = method_options["path_to_lf_data"] # necessary to substract that name from name list of all models for evaluation

        hf_data_path = method_options["path_to_hf_data"]
        lf_data_iterators = []
        for _,path in enumerate(lf_data_paths):
            lf_data_iterators.append(DataIterator(path, None, None))
        hf_data_iterator = DataIterator(hf_data_path, None, None)
        initial_design = config["method"]["initial_design"]
        global_settings = config.get('global_settings', None)

        return cls(config,lf_data_iterators, hf_data_iterator, result_description, experiment_dir, initial_design, predictive_var,BMFMC_reference,global_settings)

    def core_run(self):
        """ Generate simulation runs for lofis and hifis that cover the output
            space based on one p(y) distribution of one lofi """
        self.lf_mc_in = self.lf_data_iterators[0].read_pickle_file()[0] # here we assume that all lfs have the same input vector
        lfs_mc_out = [lf_data_iterator.read_pickle_file()[1][:,0] for _,lf_data_iterator in enumerate(self.lf_data_iterators)] # list of lf data with another list or dictionary per lf containing x_vec and y(_vec) CAREFUL: we only take the first column and omit vectorial outputs --> this should be changed!
        self.lfs_mc_out = np.atleast_2d(np.vstack(lfs_mc_out)).T

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
            exists = None #TODO delete! this is just to not write the file
            if exists:
                with open(path_to_train_data,'rb') as handle: #TODO: give better name according to experiment and choose better location
                    self.train_in, self.lfs_train_out, self.hf_train_out, self.lf_mc_in, self.lfs_mc_out = pickle.load(handle) #TODO --> Change this according to design below!!
            else:
######## TODO: THIS IS A WORKAROUND ! we calculate a subset of the mc data for the hf training -> this is usually not the case so the subset selection should be deleted!
                self.train_in, self.hf_train_out = self.hf_data_iterator.read_pickle_file()
                self.hf_train_out = self.hf_train_out[:,0] #TODO here we neglect the vectorial output --> this should be changed in the future
                self.hf_mc = self.hf_train_out
                # find the touples of lf and hf data that match
            ##### workaround starts here #####
                if self.initial_design['method'] == 'random':
                    n_bins = self.initial_design["num_bins"]
                    n_points = self.initial_design["num_HF_eval"]
                    ylf_min = np.amin(self.lfs_mc_out)
                    ylf_max = np.amax(self.lfs_mc_out)
                    break_points = np.linspace(ylf_min, ylf_max, n_bins+1)
                    bin_vec = pd.cut(self.lfs_mc_out[:,0], bins=break_points, labels=False, include_lowest=True, retbins=True) #TODO check dim of lfs_mc_out for multiple lfs
                    # Some initialization
                    self.lfs_train_out = np.empty((0,self.lfs_mc_out.shape[1])) #TODO check if this is right
                    dummy_hf = self.hf_train_out #TODO delete later
                    self.hf_train_out = np.array([]).reshape(0,1) #TODO: Later delete as HF MC is normally not available
                    self.train_in = np.array([]).reshape(0,self.lf_mc_in.shape[1])
                    # Go through all bins and select randomly select points from them (also several per bin!)
                    for bin_n in range(n_bins):
                        # array of booleans
                        y_in_bin_bool = [bin_vec[0]==bin_n]
                        # definine bin data
                        bin_data_in = self.lf_mc_in[tuple(y_in_bin_bool)]
                        bin_data_lfs = self.lfs_mc_out[tuple(y_in_bin_bool)]
                        bin_data_hf = dummy_hf[tuple(y_in_bin_bool)].reshape(-1,1) #TODO: later delete as HF MC is normally not available
                        #randomly select points in bins
                        rnd_select = [randint(0, bin_data_lfs.shape[0]-1) for p in range(n_points//n_bins)]
                        # Check if points in bin
                        if len(rnd_select) !=0:
                            #define the training data by appending the training vector
                            self.train_in = np.vstack([self.train_in,bin_data_in[rnd_select,:]])
                            self.lfs_train_out = np.vstack((self.lfs_train_out,bin_data_lfs[rnd_select,:]))
                            self.hf_train_out = np.vstack([self.hf_train_out,bin_data_hf[rnd_select,:]]) #TODO later delete as HF MC data is normally not available

            #### workaround end #############
                else: pass #TODO this needs to be changed to an keyword error check

                # write new pickle file for subsequent analysis with matching data
                with open(path_to_train_data,'wb') as handle: #TODO: give better name according to experiment and choose better location
                    pickle.dump([self.train_in, self.lfs_train_out, self.hf_train_out, self.lf_mc_in, self.lfs_mc_out],handle,protocol=pickle.HIGHEST_PROTOCOL) #TODO: check if list is right format here
        # CREATE the underlying model
        self.model = BMFMCModel.from_config_create_model(self.config, self.train_in, self.lfs_train_out, self.hf_train_out, self.lf_mc_in, self.lfs_mc_out,hf_mc=self.hf_mc) # TODO: HERE we actualluy define the BMFMC_model and bypass the input arg

####### TODO: This needs still to be implemented to run HF directly from this iterator for initial desing
        """ Run Analysis on all models """
        # here the set of new input variables will be passed to all lofis/hifi--> previous list_iterator not necessary!
 #       self.model.update_model_from_sample_batch(self.samples)

        # here all models will be evaluated (multifidelity  model will be evaluated)
        # output matrix has the form:[ ylofi1, ylofi2, ..., yhifi ]

        self.output = self.eval_model()
         # What about the already calculated points?
         #TODO some active learning here


    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()


    def active_learning(self):
        pass

    def post_run(self):
        """ Analyze the results """
        print(self.output)
        if self.result_description['plot_results']==True:

            # plt.rc('text', usetex=True)
            plt.rcParams["mathtext.fontset"] = "cm"
            plt.rcParams.update({'font.size':23})

            fig, ax = plt.subplots()
            ax.set(xlim=(0.02,0.07), ylim=(0,120))
            # plot the bmfmc var
            if self.predictive_var == "True":
               ax.plot(self.output['pyhf_support'],np.sqrt(self.output['pyhf_var']),linewidth=0.5,color='lightgreen',alpha=0.4)#,label=r'$\mathbb{SD}\left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
            # plot the bmfmc approx mean
            ax.plot(self.output['pyhf_support'],self.output['pyhf_mean'],color='xkcd:green',linewidth=1.5,label=r'$\mathbb{E}\left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
            #### plot the reference solution for classical BMFMC###
            if self.BMFMC_reference == "True":
                # plot the bmfmc var
                if self.predictive_var =="True":
                   ax.plot(self.output['pyhf_support'],np.sqrt(self.output['pyhf_var_BMFMC']),linewidth=0.5,color='magenta',alpha=0.4)#,label=r'$\mathbb{SD}\left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                # plot the bmfmc approx mean
                ax.plot(self.output['pyhf_support'],self.output['pyhf_mean_BMFMC'],color='xkcd:magenta',linewidth=1.5,label=r'$\mathbb{E}\left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]-BMFMC$')

            # plot the MC reference of HF
            ax.plot(self.output['pyhf_mc_support'],self.output['pyhf_mc'],color='k',alpha=0.8,label=r'$\mathrm{p}\left(y_{\mathrm{HF}}\right) \ \mathrm{MC \ reference}$')
            if self.lfs_mc_out.shape[1]<2:
                # plot the MC of LF
                ax.plot(self.output['pylf_mc_support'],self.output['pylf_mc'],color='r',alpha=0.7,label=r'$\mathrm{p}\left(y_{\mathrm{LF}}\right)$')

            ax.set_xlabel(r'$y$')#,usetex=True)
            ax.set_ylabel(r'$\mathrm{p}(y)$')
            ax.grid(which='major',linestyle='-')
            ax.grid(which='minor',linestyle='--',alpha=0.5)
            ax.minorticks_on()
            ax.legend()
            fig.set_size_inches(15,15)
###########################
            if self.output['manifold_test'].shape[1]<2:
                fig2, ax2 = plt.subplots()
                # plot the current dataset
                # plot the MC of LF
                ax2.plot(self.output['manifold_test'][:,0],self.output['f_mean'],linestyle='',marker='.',color='k',alpha=1,label=r'$m_{\mathrm{GP}}$')
                ax2.plot(self.output['manifold_test'][:,0],self.output['f_mean']+np.sqrt(self.output['y_var']),linestyle='',marker='.',color='grey',alpha=0.5,label=r'$\mathbb{SD}_{\mathrm{GP}}$')
                ax2.plot(self.output['manifold_test'][:,0],self.output['f_mean']-np.sqrt(self.output['y_var']),linestyle='',marker='.',color='grey',alpha=0.5)
                ax2.plot(self.output['manifold_test'][:,0],self.output['sample_mat'],linestyle='',marker='.',color='red',alpha=0.2)
                ax2.plot(self.lfs_train_out,self.hf_train_out,linestyle='',marker='x',color='r',alpha=1,label=r'Training data')
                ax2.plot(self.lfs_mc_out[:,0],self.hf_mc,linestyle='',markersize=2,marker='.',color='grey',alpha=0.7,label=r'Reference MC data')

                ax2.plot(self.hf_mc,self.hf_mc,linestyle='-',marker='',color='g',alpha=0.7,label=r'Identity')
                ax2.set_xlabel(r'$y_{\mathrm{LF}}$')#,usetex=True)
                ax2.set_ylabel(r'$y_{\mathrm{HF}}$')
                ax2.grid(which='major',linestyle='-')
                ax2.grid(which='minor',linestyle='--',alpha=0.5)
                ax2.minorticks_on()
                ax2.legend()
                fig2.set_size_inches(15,15)

            if self.output['manifold_test'].shape[1]==2:
                fig3 =plt.figure()
                ax3 = fig3.add_subplot(111,projection='3d')
                ax3.scatter(self.output['manifold_test'][:,0,None],self.output['manifold_test'][:,1,None],self.hf_mc[:,None],s=3,c='k')
                ax3.set_xlabel(r'$y_{\mathrm{LF}}$')#,usetex=True)
                ax3.set_ylabel(r'$\gamma$')
                ax3.set_zlabel(r'$y_{\mathrm{HF}}$')

    #
        ####### plot input-output scatter plot matrices ########################
#            dataset = pd.DataFrame({'$y_{\mathrm{HF}}$': self.hf_mc,'E-modulus': self.lf_mc_in[:,0], 'U-field 1': self.lf_mc_in[:,1],'U-field 2': self.lf_mc_in[:,2],'U-field 3': self.lf_mc_in[:,3],'U-field 4': self.lf_mc_in[:,4],'U-field 5': self.lf_mc_in[:,5],'U-field 6': self.lf_mc_in[:,6],'U-field 7': self.lf_mc_in[:,7],'U-field 8': self.lf_mc_in[:,8],'U-field 9': self.lf_mc_in[:,9]})
#            sns.set()
#            ma = sns.PairGrid(dataset)
#            ma = ma.map_upper(plt.scatter,s=0.1)
#            ma = ma.map_lower(sns.kdeplot, cmap="Blues_d")
#            ma = ma.map_diag(sns.kdeplot,lw=2,legend=False)
#
#            dataset = pd.DataFrame({'$y_{\mathrm{HF}}$': self.hf_mc,'latent 1': self.output['manifold_test'][:,1], 'latent 2': self.output['manifold_test'][:,2]})
#            sns.set()
#            ma = sns.PairGrid(dataset)
#            ma = ma.map_upper(plt.scatter,s=0.1)
#            ma = ma.map_lower(sns.kdeplot, cmap="Blues_d")
#            ma = ma.map_diag(sns.kdeplot,lw=2,legend=False)
#
            plt.show()




        if self.result_description['write_results']==True:
                pass



        #if self.result_description is not None:
        #    results = process_ouputs(self.output, self.result_description)
        #    write_results(results,
        #                      self.global_settings["output_dir"],
        #                      self.global_settings["experiment_name"])
        ##else:
        ##print("Size of inputs {}".format(self.samples.shape))
        ##print("Inputs {}".format(self.samples))
        #print("Size of outputs {}".format(self.output['mean'].shape))
        #print("Outputs {}".format(self.output['mean']))
