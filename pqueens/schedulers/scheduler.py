try:
    import simplejson as json
except ImportError:
    import json
import abc

import pqueens.interfaces.job_interface as job_interface
from pqueens.drivers.driver import Driver
from pqueens.utils.information_output import print_scheduling_information, print_driver_information
from pqueens.utils.manage_singularity import SingularityManager


class Scheduler(metaclass=abc.ABCMeta):
    """
    Abstract base class for schedulers in QUEENS.
    
    The scheduler manages simulation runs in QUEENS on local or remote computing resources
    with or without Singularity containers using various scheduling systems on respective
    computing resource (see also respective Wiki article for more details).

    Args:
        base_settings (dict):      dictionary containing settings from base class for
                                   further use and completion in child classes 

    Attributes:
        experiment_name (str):     name of QUEENS experiment
        polling_time (float):      polling time
        input_file (str):          path to QUEENS input file
        config (dict):             dictionary containing configuration as provided in
                                   QUEENS input file
        scheduler_type (str):      type of scheduler chosen in QUEENS input file
        experiment_dir (str):      path to QUEENS experiment directory
        restart (bool):            flag for restart
        cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                   cluster options
        ecs_task_options (dict):   (only for ECS task scheduler) further ECS task
                                   options
        remote (bool):             flag for remote scheduling
        remote connect (str):      (only for remote scheduling) adress of remote
                                   computing resource
        singularity (bool):        flag for use of Singularity containers
        port (int):                (only for remote scheduling with Singularity) port of
                                   remote resource for ssh port-forwarding to database
        restart (bool):            flag for restart
        cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                   cluster options
        ecs_task_options (dict):   (only for ECS task scheduler) further ECS task
                                   options
        job_id (int):              job ID (used for database organization)
        singularity_manager (obj): instance of Singularity-manager class
        process_ids (dict): Dict of process-IDs of the submitted process as value with job_ids as
                           keys
        driver_name (str):         Name of the driver that shall be used for job submission

    Returns:
        scheduler (obj):           instance of scheduler class

    """

    def __init__(self, base_settings):
        self.experiment_name = base_settings['experiment_name']
        self.polling_time = base_settings['polling_time']
        self.input_file = base_settings['input_file']
        self.config = base_settings['config']
        self.scheduler_type = base_settings['scheduler_type']
        self.experiment_dir = base_settings['experiment_dir']
        self.restart = base_settings['restart']
        self.cluster_options = base_settings['cluster_options']
        self.ecs_task_options = base_settings['ecs_task_options']
        self.remote = base_settings['remote']
        self.remote_connect = base_settings['remote_connect']
        self.singularity = base_settings['singularity']
        self.port = None
        self.job_id = None
        self.process_ids = {}
        self.singularity_manager = SingularityManager(
            remote=self.remote,
            remote_connect=self.remote_connect,
            singularity_bind=base_settings['cluster_options']['singularity_bind'],
            singularity_path=base_settings['cluster_options']['singularity_path'],
            input_file=self.input_file,
        )
        self.driver_name = base_settings['driver_name']

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name=None, driver_name=None):
        """
        Create scheduler from problem configuration

        Args:
            config (dict):        dictionary containing configuration
                                  as provided in QUEENS input file
            scheduler_name (str): name of scheduler
            driver_name (str): Name of driver that should be used in this job-submission
 
        Returns:
            scheduler (obj):        scheduler object

        """
        # import here to avoid issues with circular inclusion
        from .standard_scheduler import StandardScheduler
        from .cluster_scheduler import ClusterScheduler
        from .ecs_task_scheduler import ECSTaskScheduler

        scheduler_dict = {
            'standard': StandardScheduler,
            'nohup': StandardScheduler,
            'pbs': ClusterScheduler,
            'slurm': ClusterScheduler,
            'ecs_task': ECSTaskScheduler,
        }

        # get scheduler options according to chosen scheduler name
        # or without specific naming from input file
        if scheduler_name is not None:
            scheduler_options = config[scheduler_name]
        else:
            scheduler_options = config['scheduler']

        # initalize dictionary for base settings
        base_settings = {}

        # general base settings: experiment name, polling time and input file
        base_settings['experiment_name'] = config['global_settings']['experiment_name']
        base_settings['polling_time'] = config.get('polling-time', 1)
        base_settings['input_file'] = config['input_file']
        base_settings['driver_name'] = driver_name

        # base settings for scheduler:
        # 1) set complete configuration for subsequent transfer to driver
        base_settings['config'] = config

        # 2) set scheduler options as well as subsection for cluster for
        #    subsequent transfer to specific scheduler class
        base_settings['scheduler_input_options'] = scheduler_options
        if scheduler_options.get('cluster'):
            base_settings['cluster_input_options'] = scheduler_options['cluster']

        # 3) set type of scheduler
        base_settings['scheduler_type'] = scheduler_options['scheduler_type']

        # 4) set experiment directory
        base_settings['experiment_dir'] = scheduler_options['experiment_dir']

        # 5) set flag for restart from finished simulation (default: false)
        base_settings['restart'] = scheduler_options.get('restart', False)

        # 6) set flag for remote scheduling (default: false)
        #    as well as further options for remote scheduling
        if scheduler_options.get('remote'):
            base_settings['remote'] = True
            base_settings['remote_connect'] = scheduler_options['remote']['connect']
        else:
            base_settings['remote'] = False
            base_settings['remote_connect'] = None

        # 7) set flag for Singularity container (default: false) as well as
        #    subsection for subsequent transfer to specific scheduler class
        base_settings['singularity'] = scheduler_options.get('singularity', False)
        if scheduler_options.get('singularity_settings'):
            base_settings['singularity_input_options'] = scheduler_options['singularity_settings']
        else:
            base_settings['singularity_input_options'] = None

        # generate specific scheduler class
        scheduler_class = scheduler_dict[scheduler_options['scheduler_type']]
        scheduler = scheduler_class.create_scheduler_class(base_settings)

        # print out scheduling information
        print_scheduling_information(
            base_settings['scheduler_type'],
            base_settings['remote'],
            base_settings['remote_connect'],
            base_settings['singularity'],
        )

        # TODO: this might not work anymore for multiple drivers
        # print out driver information
        # (done here to print out this information only once)
        try:
            if config['driver']['driver_params'].get('post_post') is not None:
                post_post_file_prefix = config['driver']['driver_params']['post_post'].get(
                    'file_prefix'
                )
            else:
                post_post_file_prefix = None
            print_driver_information(
                config['driver']['driver_type'],
                config['driver']['driver_params'].get('cae_software_version'),
                post_post_file_prefix,
                scheduler_options.get('docker_image', None),
            )
        except KeyError:
            pass

        return scheduler

    # ------------------------ AUXILIARY HIGH LEVEL METHODS -----------------------
    def submit(self, job_id, batch):
        """ Function to submit new job to scheduling software on a given resource

        Args:
            job_id (int):            ID of job to submit
            batch (int):             Batch number of job

        Returns:
            pid (int):               process id of job

        """
        # get restart flag from job interface
        restart = job_interface.restart_flag

        if self.singularity is True:
            pid = self._submit_singularity(job_id, batch, restart)
        else:
            pid = self._submit_driver(job_id, batch, restart)

        self.process_ids[str(job_id)] = pid

        return pid

    def submit_post_post(self, job_id, batch):
        """ Function to submit new post-post job to scheduling software on a given resource

        Args:
            job_id (int):            ID of job to submit
            batch (int):             Batch number of job

       """
        # create driver
        # TODO we should not create a new driver instance here every time
        # instead only update the driver attributes.
        driver_obj = Driver.from_config_create_driver(self.config, job_id, batch, self.driver_name)

        # do post-processing (if required), post-post-processing,
        # finish and clean job
        driver_obj.post_job_run()

    # ------- CHILDREN METHODS THAT NEED TO BE IMPLEMENTED / ABSTRACTMETHODS ------
    @abc.abstractmethod
    def pre_run(self):
        pass

    @abc.abstractmethod
    def _submit_singularity(self, job_id, batch, restart):
        pass

    @abc.abstractmethod
    # check whether job is still alive (dependent on scheduler on cluster)
    def alive(self, process_id):
        pass

    @abc.abstractmethod
    def check_job_completion(self, job):
        pass

    @abc.abstractmethod
    def post_run(self):
        pass

    # ------------- private helper methods ----------------#
    def _submit_driver(self, job_id, batch, restart):
        """Submit job to driver

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            driver_obj.pid (int): process ID
        """

        # create driver
        # TODO we should not create the object here everytime!
        # TODO instead only update the attributes of the instance.
        # TODO we should specify the data base sheet as well
        driver_obj = Driver.from_config_create_driver(
            self.config, job_id, batch, self.driver_name, cluster_options=self.cluster_options
        )

        if not restart:
            # run driver and get process ID, if not restart
            driver_obj.pre_job_run_and_run_job()
            pid = driver_obj.pid

            # only required for standard scheduling: finish-and-clean call
            # (taken care of by submit_post_post for other schedulers)
            if self.scheduler_type == 'standard':
                driver_obj.post_job_run()
        else:
            # set process ID to zero as well as finish and clean, if restart
            pid = 0
            driver_obj.post_job_run()

        return pid