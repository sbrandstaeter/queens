import abc

class Scheduler(metaclass=abc.ABCMeta):
    """ Base class for schedulers """

    def __init__(self, base_settings):
        self.remote_flag = base_settings['remote_flag']
        self.json_input = base_settings['json_input']
        self.path_to_singularity = base_settings['singularity_path']

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name=None):
        """ Create scheduler from problem description

        Args:
            scheduler_name (string): Name of scheduler
            config (dict): Dictionary with QUEENS problem description

        Returns:
            scheduler: scheduler object

        """
        # import here to avoid issues with circular inclusion
        from .schedulers import Local_scheduler
        from .schedulers import PBS_scheduler
        from .schedulers import Slurm_scheduler

        scheduler_dict = {'local': LocalScheduler,
                          'pbs': PBSScheduler,
                          'slurm': SlurmScheduler}

        if scheduler_name:
            scheduler_options = config[scheduler_name]
        else:
            scheduler_options = config['scheduler']

        # determine which object to create
        scheduler_class = scheduler_dict[scheduler_options["scheduler_type"]]

########### create base settings #################################################
        if scheduler_options["scheduler_type"]=='local'
            base_settings['remote_flag'] = False
        elif scheduler_options["scheduler_type"]=='pbs' or scheduler_options["scheduler_type"]=='slurm':
            base_settings['remote_flag'] = True
        else:
            raise RuntimeError("Slurm type was not specified correctly! Choose either 'local', 'pbs' or 'slurm'!")

        base_settings['json_input'] = config['input_file']
        base_settings['singularity_path'] = config['driver']['driver_params']['path_to_singularity']
        base_settings['connect'] = config[scheduler_name]['connect_to_ressource']
########### end base settings ####################################################

        scheduler = scheduler_class.from_config_create_scheduler(config, base_settings, scheduler_name=None)
        return scheduler

#### basic init function is called in resource.py after creation of scheduler object
    def pre_run(self):
        if self.remote_flag:
            self.prepare_singularity_files()
            self.copy_temp_json()
        else:
            pass

########## Auxiliary high-level methods #############################################
    def copy_temp_json(self):
        command_list = ["scp", self.json_input, self.connect_to_resource[1:]+':'+self.path_to_singularity + '/temp.json']
        command_string = ' '.join(command_list)
        stdout,stderr = self.run_subprocess(command_string)
        if stderr:
            raise RuntimeError("Error! Was not able to copy local json input file to remote! Abort...")


    def create_singularity_image(self):
        """ Add current environment to predesigned singularity container for cluster applications """
        # create hash for files in image
        self.hash_files('hashing')
        # create the actual image
        command_string = "sudo singularity build ../utils/driver.simg ../utils/singulariy_recipe"
        _, _  = self.run_subprocess(command_string)
        print("Successfully build new singularity image. Finished! Please start Queens simulation again!")

    def hash_files(self, mode=None):
        hashlist = []
        hasher = hashlib.md5()
        # hash all drivers
        filenames = glob.glob("../drivers/")
        for filename in filenames:
            with open(filename,'rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
            hashlist.append(hasher.hexdigest())
        # hash mongodb
        with open('../database/mongodb.py','rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
        hashlist.append(hasher.hexdigest())
        # hash utils
        with open('../utils/injector.py','rb') as inputfile:
            data = inputfile.read()
            hasher.update(data)
        hashlist.append(hasher.hexdigest())
        # hash requirements_remote
        with open('../../requirements_remote.txt','rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
        hashlist.append(hasher.hexdigest())
        # hash setup_remote
        with open('../../setup_remote.py','rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
        hashlist.append(hasher.hexdigest())
        # hash remote_main
        with open('../remote_main.py','rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
        hashlist.append(hasher.hexdigest())

        # write hash list to a file in utils directory
        if mode is not None:
            hashfile = '../utils/hashfile.txt'
            with open(hashfile,'w') as f:
                f.write(hashlist)
        else:
            return hashlist


    def prepare_singularity_files(self):
        # check existence local
        if os.path.isfile('../utils/driver.simg'):
            # check singularity status local
            with open('../utils/hashfile.txt','r') as oldhash:
                old_data = oldhash.read()
            hashlist = self.hashfiles()
            # Write local singularity image
            if old_data != hashlist:
                raise Warning("Local singularity image is not up-to-date with QUEENS! Writing new local image...")
                self.create_singularity_image()
                print("Local singularity image written sucessfully!")

            # check existence singularity and hash table remote
            command_list = [self.connect_to_ressource,'test -f',self.path_to_singularity+"/driver.simg"]
            command_string = ' '.join(command_list)
            stdout,stderr = self.run_subprocess(command_string)
            if stdout is not 0: # TODO check if correct
            # Update remote image
                raise Warning("Remote singularity image is not existend! Updating remote image from local image...")
                command_list = ["scp","../utils/driver.simg",self.connect_to_resource[1:]+':'+self.path_to_singularity]
                command_string = ' '.join(command_list)
                stdout,stderr = self.run_subprocess(command_string)
                if stderr:
                    raise RuntimeError("Error! Was not able to copy local singulariy image to remote! Abort...")
            # Update remote hashfile
                command_list = ["scp","../utils/hashfile.txt",self.connect_to_resource[1:]+':'+self.path_to_singularity]
                command_string = ' '.join(command_list)
                stdout,stderr = self.run_subprocess(command_string)

            else:
            # Check remote hashfile
                print("Remote singularity image found! Checking state...")
                command_list = [self.connect_to_ressource,'cat',self.path_to_singularity+"/hashfile.txt"]
                command_string = ' '.join(command_list)
                stdout,stderr = self.run_subprocess(command_string)
                if stdout != hashlist:
                    raise Warning("Remote singularity image is not up-to-date with QUEENS! Updating remote image from local image...")
                    command_list = ["scp","../utils/driver.simg",self.connect_to_resource[1:]+':'+self.path_to_singularity]
                    command_string = ' '.join(command_list)
                    stdout,stderr = self.run_subprocess(command_string)
                    if stderr:
                        raise RuntimeError("Error! Was not able to copy local singulariy image to remote! Abort...")

            if stderr:
                raise RuntimeError("Error! Was not able to check state of remote singularity image! Abort...")
        else:
            raise Warning("No local singulariy image found! Building new image...")
            self.create_singularity_image()
            print("Local singularity image written sucessfully!")

    def run_subprocess(command_string, my_env = None):
        """ Method to run command_string outside of Python """
        if (my_env is None) and ('my_env' in mpi_config):
            p = subprocess.Popen(command_string,
                             env = self.mpi_config['my_env'],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True,
                             universal_newlines=True)
        else:
            p = subprocess.Popen(command_string,
                             env = my_env,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True,
                             universal_newlines=True)

        stdout, stderr = p.communicate()
        p.poll()
        print(stderr)
        print(stdout)
        return stdout, stderr, p #TODO if poll and return p is helpful

    def submit(self, job_id, batch):
        """ Function to submit new job to scheduling software on a given resource


        Args:
            job_id (int):               Id of job to submit
            experiment_name (string):   Name of experiment
            batch (string):             Batch number of job
            experiment_dir (string):    Directory of experiment
            database_address (string):  Address of database to connect to
            driver_options (dict):      Options for driver

        Returns:
            int: proccess id of job

        """
        remote_args_list = '--job_id={} --batch={}'.format(job_id, batch)
        remote_args = ' '.join(remote_args_list)
        cmdlist_remote_main = [self.connect_to_ressource, "." + self.path_to_singularity + "/driver.simg", remote_args]
        cmd_remote_main = ' '.join(cmdlist_remote_main)
        stdout, stderr, p = self.run_subprocess(cmd_remote_main)

        if stderr:
            raise RuntimeError("The file 'remote_main' in remote singularity image could not be executed properly!")
            print(stderr)

        # get the process id from text output
        match = self.get_process_id_from_output(stdout)
        try:
            return int(match)
        except:
            sys.stderr.write(output)
            return None

######### Children methods that need to be implemented / abstractmethods ######################
    @abc.abstractmethod # how to check this is dependent on cluster / env
    def alive(self,process_id):
        pass
