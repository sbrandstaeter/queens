import abc
import os
import glob
from pqueens.utils.run_subprocess import run_subprocess


class PostPost(metaclass=abc.ABCMeta):
    """ Base class for post post processing

        Attributes:
            delete_data_flag (bool):  Delete files after processing
            post_post_file_name_prefix_lst (lst): List with prefixes of post-processed files
            error ():
            output_dir (str): Path to result files
            result (np.array): Array containing the quantities of interest

    """

    def __init__(self, delete_data_flag, post_post_file_name_prefix_lst):
        """ Init post post class

            Args:
                delete_data_flag (bool): Delete files after processing
                post_post_file_name_prefix_lst (lst): List with prefixes of result files

        """

        self.delete_data_flag = delete_data_flag
        self.post_post_file_name_prefix_lst = post_post_file_name_prefix_lst

        self.error = False
        self.output_dir = None
        self.result = None

    @classmethod
    def from_config_create_post_post(cls, config):
        """
        Create PostPost object from problem description

        Args:
            config (dict): input json file with problem description

        Returns:
            post_post: post_post object

        """

        from .post_post_ansys import PostPostANSYS
        from .post_post_baci import PostPostBACI
        from .post_post_deal import PostPostDEAL
        from .post_post_generic import PostPostGeneric
        from .post_post_openfoam import PostPostOpenFOAM
        from .post_post_baci_shape import PostPostBACIShape
        from .post_post_net_cdf import PostPostNetCDF
        from .post_post_baci_vectorized import PostPostBACIVector
        from .post_post_baci_ensight import PostPostBACIEnsight

        post_post_dict = {
            'ansys': PostPostANSYS,
            'baci': PostPostBACI,
            'baci_vector': PostPostBACIVector,
            'baci_ensight': PostPostBACIEnsight,
            'deal': PostPostDEAL,
            'generic': PostPostGeneric,
            'openfoam': PostPostOpenFOAM,
            'baci_shape': PostPostBACIShape,
            'netCDF': PostPostNetCDF,
        }

        # determine which object to create
        # TODO this is not a reliable approach? What if we have multiple drivers?
        # However, this cannot be fixed by itself here, but we need to
        # cleanup the whole input parameter handling to fix this.
        post_post_options = config['driver']['driver_params']['post_post']

        if post_post_options['post_post_approach_sel'] == 'ansys':
            post_post_version = 'ansys'
        elif post_post_options['post_post_approach_sel'] == 'baci':
            post_post_version = 'baci'
        elif post_post_options['post_post_approach_sel'] == 'baci_ensight':
            post_post_version = 'baci_ensight'
        elif post_post_options['post_post_approach_sel'] == 'baci_vector':
            post_post_version = 'baci_vector'
        elif post_post_options['post_post_approach_sel'] == 'deal':
            post_post_version = 'deal'
        elif post_post_options['post_post_approach_sel'] == 'generic':
            post_post_version = 'generic'
        elif post_post_options['post_post_approach_sel'] == 'baci_shape':
            post_post_version = 'baci_shape'
        elif post_post_options['post_post_approach_sel'] == 'openfoam':
            post_post_version = 'openfoam'
        elif post_post_options['post_post_approach_sel'] == 'netCDF':
            post_post_version = 'netCDF'
        else:
            raise RuntimeError("post_post_approach_sel not set, fix your input file")

        post_post_class = post_post_dict[post_post_version]

        # ---------------------------- CREATE BASE SETTINGS ---------------------------
        base_settings = {}
        base_settings['options'] = post_post_options

        # overwrite database settings so that we dont delete the existing dbs
        config['database']['reset_database'] = False
        config['database']['drop_all_existing_dbs'] = False

        base_settings['config'] = config
        post_post = post_post_class.from_config_create_post_post(base_settings)
        return post_post

    def copy_post_files(self, files_of_interest, remote_connect, remote_output_dir):
        """ Copy identified post-processed files from "local" output directory
            to "remote" output directory in case of remote scheduling """

        remote_file_name = os.path.join(remote_output_dir, '.')
        command_list = [
            "scp ",
            remote_connect,
            ":",
            files_of_interest,
            " ",
            remote_file_name,
        ]
        command_string = ''.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)

        # detection of failed command
        if stderr:
            raise RuntimeError(
                "\nPost-processed file could not be copied from remote machine!"
                f"\nStderr:\n{stderr}"
            )

    def error_handling(self, output_dir):
        """ Mark failed simulation and set results appropriately

            What does this function do?? This is super unclear
        """
        # TODO  ### Error Types ###
        # No QoI file
        # Time/Time step not reached
        # Unexpected values

        # Organized failed files
        input_file_extention = 'dat'
        # TODO add documentation what happens here?
        # TODO Make this platform independent
        if self.error is True:
            command_string = (
                "cd "
                + output_dir
                + "&& cd ../.. && mkdir -p postpost_error && cd "
                + output_dir
                + r"&& cd .. && mv *."
                + input_file_extention
                + r" ../postpost_error/"
            )
            _, _, _, _ = run_subprocess(command_string)

    def delete_field_data(self, output_dir, remote_connect):
        """ Delete output files except files with given prefix """

        inverse_prefix_expr = r"*[!" + self.post_post_file_name_prefix_lst + r"]*"
        files_of_interest = os.path.join(output_dir, inverse_prefix_expr)
        post_file_list = glob.glob(files_of_interest)

        for filename in post_file_list:
            if remote_connect is not None:
                # generate command for removing 'post-processed file
                # from output directory on remote machine
                command_list = [
                    'ssh',
                    remote_connect,
                    '"rm',
                    filename,
                    '"',
                ]
                command_string = ' '.join(command_list)
                _, _, _, stderr = run_subprocess(command_string)

                # detection of failed command
                if stderr:
                    raise RuntimeError(
                        "\nPost-processed file could not be removed from remote machine!"
                        f"\nStderr on remote:\n{stderr}"
                    )
            else:
                os.remove(filename)

    def postpost_main(self, local_output_dir, remote_connect, remote_output_dir):
        """ Main routine for managing post-post-processing

            Args:
                local_output_dir (str): Path to "local" output directory
                remote_connect (str):  address of remote computing resource (only for remote
                scheduling)
                remote_output_dir (str): Path to "remote" output directory
                (from the point of view of the location of the post-processed files)

            Returns:
                result (np.array): Result of the post-post operation which is the current value
                                   of the quantities of interest

        """

        # identify post-processed files containing data of interest in "local" output directory
        prefix_expr = []
        files_of_interest = []
        for file_prefix in self.post_post_file_name_prefix_lst:
            prefix_expr.append('*' + file_prefix + '*')
            files_of_interest.append(os.path.join(local_output_dir, prefix_expr[-1]))

        if remote_connect is not None:
            # copy identified post-processed files from "local" output directory
            # to "remote" output directory in case of remote scheduling
            for file in files_of_interest:
                self.copy_post_files(file, remote_connect, remote_output_dir)

            # set output directory to "remote"
            output_dir = remote_output_dir
        else:
            # set output directory to "local"
            output_dir = local_output_dir

        # get data of interest from identified post-processed files in output directory
        for current_idx, current_file in enumerate(files_of_interest):
            self.read_post_files(current_file, idx=current_idx)

        # mark failed simulation and set results appropriately in output directory
        self.error_handling(output_dir)

        # clear memory by removing all other post-processed files in either "local"
        # or "remote" output directory
        if self.delete_data_flag is True:
            self.delete_field_data(output_dir, remote_connect)

        return self.result

    @abc.abstractmethod
    def read_post_files(self, file_names, **kwargs):
        """ This method has to be implemented by all child classes """
        pass
