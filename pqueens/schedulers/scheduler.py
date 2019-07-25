import abc
import os.path
import hashlib
import os
import subprocess
import pdb

class Scheduler(metaclass=abc.ABCMeta):
    """ Base class for schedulers """

    def __init__(self, base_settings):
        self.remote_flag = base_settings['remote_flag']
        self.json_input = base_settings['json_input']
        self.path_to_singularity = base_settings['singularity_path']
        self.connect_to_resource = base_settings['connect']

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
        from .local_scheduler import LocalScheduler
        from .PBS_scheduler import PBSScheduler
        from .slurm_scheduler import SlurmScheduler

        scheduler_dict = {'local': LocalScheduler,
                          'pbs': PBSScheduler,
                          'slurm': SlurmScheduler}

        if scheduler_name is not None:
            scheduler_options = config[scheduler_name]
        else:
            scheduler_options = config['scheduler']

        # determine which object to create
        scheduler_class = scheduler_dict[scheduler_options["scheduler_type"]]

########### create base settings #################################################
        base_settings={} # initialize empty dictionary
        if scheduler_options["scheduler_type"]=='local':
            base_settings['remote_flag'] = False
        elif scheduler_options["scheduler_type"]=='pbs' or scheduler_options["scheduler_type"]=='slurm':
            base_settings['remote_flag'] = True
        else:
            raise RuntimeError("Slurm type was not specified correctly! Choose either 'local', 'pbs' or 'slurm'!")

        base_settings['json_input'] = config['input_file']
        base_settings['singularity_path'] = config['driver']['driver_params']['path_to_singularity']
        base_settings['connect'] = config[scheduler_name]['connect_to_resource']
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
        command_list = ["scp", self.json_input, self.connect_to_resource+':'+self.path_to_singularity + '/temp.json']
        command_string = ' '.join(command_list)
        stdout,stderr,_ = self.run_subprocess(command_string)
        if stderr:
            raise RuntimeError("Error! Was not able to copy local json input file to remote! Abort...")


    def create_singularity_image(self):
        """ Add current environment to predesigned singularity container for cluster applications """
        # create hash for files in image
        self.hash_files('hashing')
        # create the actual image
        script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in
        rel_path1 ='../../driver.simg'
        rel_path2 ='../../singularity_recipe'
        abs_path1 = os.path.join(script_dir,rel_path1)
        abs_path2 = os.path.join(script_dir,rel_path2)
        command_list = ["sudo singularity build",abs_path1,abs_path2]
        command_string = ' '.join(command_list)
        _, stderr, _ = self.run_subprocess(command_string)

    def hash_files(self, mode=None):
        hashlist = []
        hasher = hashlib.md5()
        # hash all drivers
        script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in
        rel_path = "../drivers"
        abs_path = os.path.join(script_dir,rel_path)
        elements = os.listdir(abs_path)
        filenames = [os.path.join(abs_path,ele) for _,ele in enumerate(elements) if ele.endswith('.py')]
        for filename in filenames:
            with open(filename,'rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
            hashlist.append(hasher.hexdigest())
        # hash mongodb
        #TODO continue here with absolute path
        rel_path ="../database/mongodb.py"
        abs_path = os.path.join(script_dir,rel_path)
        with open(abs_path,'rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
        hashlist.append(hasher.hexdigest())
        # hash utils
        rel_path ='../utils/injector.py'
        abs_path = os.path.join(script_dir,rel_path)
        with open(abs_path,'rb') as inputfile:
            data = inputfile.read()
            hasher.update(data)
        hashlist.append(hasher.hexdigest())
        # hash requirements_remote
        rel_path='../../requirements_remote.txt'
        abs_path = os.path.join(script_dir,rel_path)
        with open(abs_path,'rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
        hashlist.append(hasher.hexdigest())
        # hash setup_remote
        rel_path='../../setup_remote.py'
        abs_path = os.path.join(script_dir,rel_path)
        with open(abs_path,'rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
        hashlist.append(hasher.hexdigest())
        # hash remote_main
        rel_path='../remote_main.py'
        abs_path = os.path.join(script_dir,rel_path)
        with open(abs_path,'rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
        hashlist.append(hasher.hexdigest())
        # hash postpost files
        rel_path='../post_post/post_post.py'
        abs_path = os.path.join(script_dir,rel_path)
        with open(abs_path,'rb') as inputfile:
                data = inputfile.read()
                hasher.update(data)
        hashlist.append(hasher.hexdigest())


        # write hash list to a file in utils directory
        if mode is not None:
            rel_path = '../../hashfile.txt'
            abs_path = os.path.join(script_dir,rel_path)
            with open(abs_path,'w') as f:
                for item in hashlist:
                    f.write("%s" % item)
        else:
            return hashlist


    def prepare_singularity_files(self):
        # check existence local
        script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in
        rel_path ='../../driver.simg'
        abs_path = os.path.join(script_dir,rel_path)
        if os.path.isfile(abs_path):
            # check singularity status local
            rel_path ='../../hashfile.txt'
            abs_path = os.path.join(script_dir,rel_path)
            with open(abs_path,'r') as oldhash:
                old_data = oldhash.read()
            hashlist = self.hash_files()
            # Write local singularity image
            if old_data != ''.join(hashlist):
                print("Local singularity image is not up-to-date with QUEENS! Writing new local image...")
                # deleting old image
                rel_path = '../../driver.simg'
                abs_path = os.path.join(script_dir,rel_path)
                command_list = ['rm',abs_path]
                command_string = ' '.join(command_list)
                _,_,_ = self.run_subprocess(command_string)
                self.create_singularity_image()
                print("Local singularity image written sucessfully!")

            # check existence singularity and hash table remote
            command_list = ['ssh -T',self.connect_to_resource,'test -f',self.path_to_singularity+"/driver.simg && echo 'Y' || echo 'N'"]
            command_string = ' '.join(command_list)
            stdout,stderr,_ = self.run_subprocess(command_string)
            if 'N' in stdout: # TODO check if correct
            # Update remote image
                print("Remote singularity image is not existend! Updating remote image from local image...")
                rel_path ="../../driver.simg"
                abs_path = os.path.join(script_dir,rel_path)
                command_list = ["scp",abs_path,self.connect_to_resource+':'+self.path_to_singularity]
                command_string = ' '.join(command_list)
                stdout,stderr,_ = self.run_subprocess(command_string)
                if stderr:
                    raise RuntimeError("Error! Was not able to copy local singulariy image to remote! Abort...")
            # Update remote hashfile
                rel_path ="../../hashfile.txt"
                abs_path = os.path.join(script_dir,rel_path)
                command_list = ["scp",abs_path,self.connect_to_resource+':'+self.path_to_singularity]
                command_string = ' '.join(command_list)
                stdout,stderr,_ = self.run_subprocess(command_string)

            elif 'Y' in stdout:
            # Check remote hashfile
                print("Remote singularity image found! Checking state...")
                command_list = ['ssh',self.connect_to_ressource, 'cat',self.path_to_singularity+"/hashfile.txt"]
                command_string = ' '.join(command_list)
                stdout,stderr,_ = self.run_subprocess(command_string)
                if stdout != ''.join(hashlist):
                    print("Remote singularity image is not up-to-date with QUEENS! Updating remote image from local image...")
                    rel_path = "../../driver.simg"
                    abs_path = os.path.join(script_dir,rel_path)
                    command_list = ["scp",abs_path,self.connect_to_resource+':'+self.path_to_singularity]
                    command_string = ' '.join(command_list)
                    stdout,stderr,_ = self.run_subprocess(command_string)
                    if stderr:
                        raise RuntimeError("Error! Was not able to copy local singulariy image to remote! Abort...")

            if stderr:
                raise RuntimeError("Error! Was not able to check state of remote singularity image! Abort...")
            print('All singularity images ok! Starting simulation on cluster...')
        else:
            print("No local singularity image found! Building new image...")
            self.create_singularity_image()
            print("Local singularity image written sucessfully!")

    def run_subprocess(self,command_string):
        """ Method to run command_string outside of Python """
        p = subprocess.Popen(command_string,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True,
                             universal_newlines=True)

        stdout, stderr = p.communicate()
        p.poll()
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
        remote_args = '--job_id={} --batch={}'.format(job_id, batch)
        cmdlist_remote_main = ['ssh',self.connect_to_ressource, "." + self.path_to_singularity + "/driver.simg", remote_args]
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
