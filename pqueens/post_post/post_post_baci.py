import glob
from io import StringIO
import os
import numpy as np
import pandas as pd
from pqueens.post_post.post_post import PostPost


class PostPostBACI(PostPost):
    """ Class for post-post-processing BACI output
    
    TODO complete docstring 
        Attributes:
            num_post ():
            time_tol ():
            target_time ():
            skiprows ():

    """

    def __init__(self, num_post, time_tol, target_time, skiprows,
                 usecols, delete_data_flag, file_prefix):
        """ Init object


        TODO complete docstring 

        Args:
            num_post ():
            time_tol ():
            target_time ():
            skiprows ():
            usecols ():
            delete_data_flag ():
            file_prefix ():

        """

        super(PostPostBACI, self).__init__(usecols, delete_data_flag, file_prefix)

        self.num_post = num_post
        self.time_tol = time_tol
        self.target_time = target_time
        self.skiprows = skiprows

    @classmethod
    def from_config_create_post_post(cls, config, base_settings):
        """ Create post_post routine from problem description

        Args:
            config (dict): input json file with problem description
            base_settings (dict): TODO what is this?? why are there two dicts?

        Returns:
            post_post: PostPostBACI object
        """
        post_post_options = base_settings['options']

        num_post = len(config['driver']['driver_params']['post_process_options'])
        time_tol = post_post_options['time_tol']
        target_time = post_post_options['target_time']
        skiprows = post_post_options['skiprows']
        usecols = post_post_options['usecols']
        delete_data_flag = post_post_options['delete_field_data']
        file_prefix = post_post_options['file_prefix']

        return cls(num_post, time_tol, target_time, skiprows,
                   usecols, delete_data_flag, file_prefix)

    # ------------------------ COMPULSORY CHILDREN METHODS ------------------------
    def read_post_files(self):
        """ Loop over post files in given output directory """

        prefix_expr = '*' + self.file_prefix + '*'
        files_of_interest = os.path.join(self.output_dir, prefix_expr)
        post_files_list = glob.glob(files_of_interest)
        post_out = []

        for filename in post_files_list:
            try:
                post_data = pd.read_csv(
                    filename,
                    sep=r',|\s+',
                    usecols=self.usecols,
                    skiprows=self.skiprows,
                    engine='python',
                )
                identifier = abs(post_data.iloc[:, 0] - self.target_time) < self.time_tol
                quantity_of_interest = post_data.loc[identifier].iloc[0, 1]
                post_out = np.append(post_out, quantity_of_interest)
                # select only row with timestep equal to target time step
                # TODO add documentation what does this do? Why is it done this way?
                if not post_out:  # timestep reached? <=> variable is empty?
                    self.error = True
                    self.result = None
                    break
            except IOError:
                self.error = True  # TODO in the future specify which error type
                self.result = None
                break
        self.error = False
        self.result = post_out
