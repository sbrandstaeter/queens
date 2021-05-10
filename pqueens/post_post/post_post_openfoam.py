import glob
from io import StringIO
import os
import numpy as np
import pandas as pd
from pqueens.post_post.post_post import PostPost


class PostPostOpenFOAM(PostPost):
    """ Class for post-post-processing OpenFOAM output

        Attributes:
            time_tol_lst (lst):     List with tolerances if desired time can not be matched
                                    exactly. Entries relate to files that are post-post processed
            target_time_lst (lst):  Time at which to evaluate QoI
            skip_rows_lst (lst):    List with number of header rows to skip per post-processed file

    """

    def __init__(
        self,
        time_tol_lst,
        target_time_lst,
        skip_rows_lst,
        use_cols_lst,
        delete_data_flag,
        post_post_file_name_prefix_lst,
    ):
        """ Init PostPost object

        Args:
            time_tol_lst (lst):    List with tolerances if desired time can not be matched
                                     exactly. Entries relate to files that are post-post  processed
            target_time_lst (lst): List with time at which to evaluate QoI per respective
                                     post-file
            skip_rows_lst (lst):      List with number of header rows to skip per post-file
            use_cols_lst (list):      List with indices of columns to use in result file. List
                                      entry corresponds to files in post_post_file_name_prefix_lst
            delete_data_flag (bool):  Delete files after processing
            post_post_file_name_prefix_lst (lst): List with prefixes of result files

        """

        super(PostPostOpenFOAM, self).__init__(delete_data_flag, post_post_file_name_prefix_lst)
        self.use_col_lst = use_cols_lst
        self.time_tol_lst = time_tol_lst
        self.target_time_lst = target_time_lst
        self.skip_rows_lst = skip_rows_lst

    @classmethod
    def from_config_create_post_post(cls, options):
        """ Create post_post routine from problem description

        Args:
            options (dict): input options

        Returns:
            post_post: PostPostOpenFOAM object
        """
        post_post_options = options['options']

        time_tol_lst = post_post_options.get('time_tol_lst')
        assert isinstance(
            time_tol_lst, (list, type(None))
        ), "The option time_tol_lst must be of type list!"

        target_time_lst = post_post_options['target_time_lst']
        assert isinstance(target_time_lst, list), "The option target_time_lst must be of type list!"

        skip_rows_lst = post_post_options['skip_rows_lst']
        assert isinstance(skip_rows_lst, list), "The option skip_rows_lst must be of type list!"

        use_col_lst = post_post_options['use_col_lst']
        assert isinstance(use_col_lst, list), "The option use_col_lst must be of type list!"

        delete_data_flag = post_post_options['delete_field_data']

        post_post_file_name_prefix_lst = post_post_options['post_post_file_name_prefix_lst']
        assert isinstance(
            post_post_file_name_prefix_lst, list
        ), "The option post_post_file_name_prefix_lst must be of type list!"

        return cls(
            time_tol_lst,
            target_time_lst,
            skip_rows_lst,
            use_col_lst,
            delete_data_flag,
            post_post_file_name_prefix_lst,
        )

    def read_post_files(self, file_names, **kwargs):
        """
        Loop over post files in given output directory

        Args:
            file_names (str): Path with filenames without specific extension

        Returns:
            None

        """
        # get index
        idx = kwargs.get('idx')

        # recover prefix and path to output directory
        output_dir, prefix_expr = os.path.split(file_names)

        # set path to time directory and get files of interest
        time_dir = os.path.join(output_dir, str(self.target_time_lst[idx]))
        file_names = os.path.join(time_dir, prefix_expr)

        post_files_list = glob.glob(file_names)
        # glob returns arbitrary list -> need to sort the list before using
        post_files_list.sort()
        post_out = np.empty(shape=0)

        for filename in post_files_list:
            try:
                post_data = pd.read_csv(
                    filename,
                    sep=r',|\s+',
                    usecols=self.use_col_lst[idx],
                    skiprows=self.skip_rows_lst[idx],
                    engine='python',
                )
            except IOError:
                _logger.info("Could not read csv-file.")
                self.error = True
                self.result = None
                break

            data_extract = str(post_data.iloc[0, 0])
            if "(" in data_extract:
                final_data_extract = data_extract.replace('(', '')
            elif ")" in data_extract:
                final_data_extract = data_extract.replace(')', '')
            else:
                final_data_extract = data_extract
            quantity_of_interest = float(final_data_extract)
            post_out = np.append(post_out, quantity_of_interest)

        self.error = False
        if self.result is None:
            self.result = post_out
        else:
            self.result = np.append(self.result, post_out)
