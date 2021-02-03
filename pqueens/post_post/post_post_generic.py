import glob

import numpy as np
import pandas as pd
from pqueens.post_post.post_post import PostPost


class PostPostGeneric(PostPost):
    """ Class for post-post-processing output from a csv file

        Attributes:
            use_col_lst (lst):   List of lists (per result file path) with columns to extract
            use_row_lst (lst):  List of lists (per result file path) with Index of rows to extract

    """

    def __init__(
        self,
        skip_rows_lst,
        use_col_lst,
        use_row_lst,
        delete_data_flag,
        post_post_file_name_prefix_lst,
    ):
        """ Init PostPost object

        Args:
            skip_rows_lst (lst):          List with number of header rows to skip per file prefix
            use_col_lst (lst):            List with indices of row to extract per file prefix
            use_row_lst (lst):            List with indices of column to extract per file prefix
            delete_data_flag (bool):      Delete files after processing
            post_post_file_name_prefix_lst (lst):        List with prefixes of result files

        """
        super(PostPostGeneric, self).__init__(delete_data_flag, post_post_file_name_prefix_lst)

        self.skip_rows_lst = skip_rows_lst
        self.use_col_lst = use_col_lst
        self.use_row_lst = use_row_lst

    @classmethod
    def from_config_create_post_post(cls, options):
        """ Create post_post routine from problem description

        Args:
            options (dict): input options

        Returns:
            post_post: PostPostGeneric object
        """
        post_post_options = options['options']

        use_col_lst = post_post_options['use_col_lst']
        assert isinstance(use_col_lst, list), "The option use_col_lst must be of type list!"

        use_row_lst = post_post_options['use_row_lst']
        assert isinstance(use_row_lst, list), "The option use_row_lst must be of type list!"

        skip_rows_lst = post_post_options['skiprows']
        assert isinstance(skip_rows_lst, list), "The option skip_rows_lst must be of type list!"

        delete_data_flag = post_post_options['delete_field_data']

        post_post_file_name_prefix_lst = post_post_options['post_post_file_name_prefix_lst']
        assert isinstance(post_post_file_name_prefix_lst, list), (
            "The option " "post_post_file_name_prefix_lst " "must be of type list!"
        )

        return cls(
            skip_rows_lst,
            use_col_lst,
            use_row_lst,
            delete_data_flag,
            post_post_file_name_prefix_lst,
        )

    def read_post_files(self, file_name, current_file_name_num):
        """
        Loop over post files in given output directory

        Args:
            file_name (str): Path with filename without specific extension
            current_file_name_num (int): Number of current filename path

        Returns:
            None

        """
        post_files_list = glob.glob(file_name)
        # glob returns arbitrary list -> need to sort the list before using
        post_files_list.sort()
        post_out = np.empty(shape=0)

        for filename in post_files_list:
            try:
                post_data = pd.read_csv(
                    filename,
                    sep=r',|\s+',
                    usecols=self.use_col_lst[current_file_name_num],
                    skiprows=self.skip_rows_lst[current_file_name_num],
                    engine='python',
                )
                quantity_of_interest = post_data.loc[self.use_row_lst[current_file_name_num]]
                post_out = np.append(post_out, quantity_of_interest)
                # very simple error check
                # TODO the error check should be done more selective so that we know which part
                #  of the result did actually fail
                if not np.any(post_out):
                    self.error = True
                    self.result = None
                    break
            except IOError:
                self.error = True
                self.result = None
                break

        self.error = False

        if self.result is None:
            self.result = post_out
        else:
            self.result = np.append(self.result, post_out)
