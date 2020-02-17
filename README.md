# QUEENS
This repository contains the code of *QUEENS*. A general purpose framework for Uncertainty Quantification,
Physics-Informed Machine Learning, Bayesian Optimization, Inverse Problems and Simulation Analytics on distributed computer 
systems.

## Content
1. [Installation and setup](#installation-and-setup)
   1. [Setup of MongoDB](#setup-of-mongodb)
      1. [Configuration of Git ](#configuration-of-git)
      1. [Clone and install QUEENS](#clone-the-queens-repository-and-install-queens)
      1. [Setup of the database](#setup-of-the-database)
         1. [Installation of MongoDB](#installation-of-mongodb)
         1. [Starting MongoDB](#starting-mongodb)
         1. [Configure MongoDB](#configure-the-database)
   1. [Remote computing](#remote-computing)
   1. [LNM-specific issues](#lnm-specific-issues)
1. [Building the documentation](#building-the-documentation)
1. [Run the test suite](#run-the-test-suite)
1. [Additional information](#additional-information)
    1. [GitLab testing machine](#make-your-machine-a-gitlab-testing-machine)
    1. [Anaconda tips and tricks](#anaconda-tips-and-tricks)
    1. [Kill open ssh-ports](#kill-open-ssh-ports-of-crashed-simulations)

## Installation and setup
*QUEENS* needs only to be installed on the localhost (e.g., *bohr*). Computations on distributed systems
as high-performance clusters (HPC) or computation-pools are automatically managed via 
[Singularity images](https://singularity.lbl.gov/).
While no Python installation is required on remote machines, the support of Singularity is necessary. For LNM,
the latter is the case for the clusters *Bruteforce* and *Deep*.

[↑ Contents](#contents)
### Configuration of Git 
> **Note** this can be skipped if you have already configured Git for other projects

A Git version >= 2.9 is required. <!-- We need at least this version to be able to configure the path to the git-hooks as outlined below. -->
Consult the official [Git documentation](www.git-scm.org) to obtain a more recent Git installation if necessary.

1. Set your username to your full name, i.e., first name followed by last name,
and your email address to your institute email address with the following commands:

    ```bash
    git config --global user.name "<Firstname> <Lastname>"
    git config --global user.email <instituteEmailAddress>
    ```

1. Set a default text editor that will be used whenever you need to write a message in Git. To set `vim` as your default text editor, type:

    ```bash
    git config --global core.editor vim
    ```

    > **Note:** Another popular choice is `kwrite`.

1. Set path to our common set of `git-hooks`. After [cloning the repository](#clone-the-repository) into the directory `<someBaseDir>/<sourceDir>`, run

    ```bash
    cd <someBaseDir>/<sourceDir>
    git config core.hooksPath ./utilities/code_checks/
    ```
   
1. In case you are using GitLab for the first time on your machine: Add your public SSH key to your GitLab 
user profile under the section `User settings - SSH keys`
    1. Check for an existing SSH key pair using: 
        ```bash 
        cat ~/.ssh/id_rsa.pub
        ```
    1. In case of an empty response, a new key pair can be generated using:
        ```bash
        ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
        ```
    1. For further instruction on how to configure your SSH key correctly please have a look at the
     [GitLab documentation](https://gitlab.lrz.de/help/ssh/README).
     
[↑ Contents](#contents)
### Clone the QUEENS repository and install QUEENS
1. You can then clone this repository to your local machine with  
`git clone git@gitlab.lrz.de:jbi/queens.git <target-directory>`  
1. [Install](http://docs.anaconda.com/anaconda/install/linux/) the latest version of 
[Anaconda](https://www.anaconda.com/) with Python 3.x.
 *Anaconda* is an open-source Python distribution and provides a
 [virtual environment manager named *Conda*](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) with many popular data science Python packages. 1. After setting up Anaconda on your machine, create a new, dedicated QUEENS development environment via  
`conda create -n <name_of_new_environment> python=3.8`
1. You need to activate the newly created environment via  
`conda activate <name_of_new_environment>`
1. All required third party libraries can then be simply installed in the environment by running:  
`pip install --upgrade -r <your-path-to-QUEENS>/requirements.txt`
1. Now, QUEENS can be installed in the environment using:  
`python <your-path-to-QUEENS>/setup.py develop`  
If you encounter any problems try using the --user flag.


**Update your Python packages from time to time using (with active conda QUEENS environment):**<br />
```bash
conda update --all && pip install --upgrade -r <your-path-to-QUEENS>/requirements.txt 
```  

**Update Python version of your conda environment:**<br />
Note: `conda update --all` updates all packages but not Python itself.  
```bash
cd <your-path-to-QUEENS>
conda activate <your-QUEENS-env>
conda update python 
conda update --all 
pip install --upgrade -r requirements.txt 
python setup.py develop
```  

**Use conda environments in Jupyter notebooks:**<br />
To get your conda environments into your Jupyter kernel, run:  
`conda install nb_conda_kernels`

**Uninstall:**<br />
To uninstall *QUEENS* from your conda environment, run  
`python setup.py develop --uninstall`  
within the activated environment.

[↑ Contents](#contents)
### Setup of the database
QUEENS writes results into a [MongoDB database](https://www.mongodb.com/), therefore QUEENS requires certain
 write and access rights. MongoDB does not necessarily have to run on the same machine as QUEENS. 
 In certain situations, it makes sense to have the database running on a different computer and connect to the 
 database via port-forwarding.

[↑ Contents](#contents)
####  Installation of MongoDB
We strongly recommend the use of MongoDB version 3.4 as this is proven to work in the current framework.

- Installation instructions if you are running **macOS** can be found [here](https://docs.mongodb.com/master/tutorial/install-mongodb-on-os-x/?_ga=2.181134695.1149150790.1494232459-1730069423.1494232449)
- Installation instructions for LNM workstations running **CentOS 7** can be done directly from the **CentOS 7**
by running:
    ```
    sudo yum install mongodb-org
    ```
- Installation instructions for LNM workstations running **Fedora 22** can be found [here](https://blog.bensoer.com/install-mongodb-3-0-on-fedora-22/)

[↑ Contents](#contents)
#### Starting MongoDB
After installation, you need to start MongoDB.
1. For LNM machines this requires that you can execute the commands:  
    ```bash 
    sudo systemctl start mongod
    ``` 
1. (Optional) In case you encounter any problems the database can be:
    1. stopped using:
        ```bash
        sudo systemctl stop mongod
       ```  
    1. or restarted using:
        ```bash
        sudo systemctl restart mongod
        ```  
1. (Optional) On a Mac you have to run: 
    ```bash 
    mongod --dbpath <path_to_your_database>`
    ```
    > Note: It could be that you have to edit the sudoers file `/etc/sudoers.d/` together with your administrator in order get the execution rights.

[↑ Contents](#contents)
#### Configure the database
Edit the MongoDB config file such that it allows connections from anywhere (probably not the safest option, but ok for now).
If you followed the standard installation instructions, you should edit the config file using  
```bash
sudo vim /etc/mongod.conf  
```
And comment out the `bindIp` like shown
```shell
# network interfaces
net:
port: 27017
# bindIp: 127.0.0.1  <-- comment out this line
```
>Note: The machine running the MongoDB database does not need to be the same as either the
machine running QUEENS or the compute cluster. However, the other machines need to be
able to connect to the MongoDB via tcp in order to write data. For that to work
within the LNM network, all of the machines need to be connected to the internal network.

[↑ Contents](#contents)
### Remote computing
Remote computing is solved via `ssh port forwarding` and `singularity images`. Please make sure you either have an existing
singularity image `driver.simg` or you have `singularity` installed on the localhost. 
In CentOs7, singularity can directly be installed from the repository via:
```bash
sudo yum install singularity
```
Connecting via ssh to the compute machine and from the compute machine to the localhost needs to work passwordless. 
Therefore, we need to copy the respective `id_rsa.pub`-keys on the localhost and the remote, once.
Firstly, please copy the localhost's `id_rsa.pub` -key to the `authorized_keys` file on the remote, 
while being logged-in on the localhost, using:
```bash
cat ~/.ssh/id_rsa.pub | ssh <username>@remote 'cat >> ~/.ssh/authorized_keys'  
```
Secondly, log-in onto the remote machine and then repeat copy the remote's `id_rsa.pub` -key to the `authorized_keys` file 
on the localhost, using:
```bash
cat ~/.ssh/id_rsa.pub | ssh <username>@localmachine 'cat >> ~/.ssh/authorized_keys'  
```
In case you do not have a `id_rsa.pub`-key on the remote machine, you can generate the key by running 
the subsequent command on the remote machine:
```batch
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
```
To learn more about how ssh-port forwarding works click 
[here](https://chamibuddhika.wordpress.com/2012/03/21/ssh-tunnelling-explained/).

[↑ Contents](#contents) 
### LNM specific issues
In case you want to run simulations on remote computing machines (e.g., cluster), you need enable access 
from the remote to the localhost at `port 27017`. By default, the fire wall software `firewalld` blocks every incoming 
request. Hence, to enable a connections, we have 
add so called rules to `firewalld` in order to connect to the database.
1.  First check the current firewall rules by typing:
    ```bash 
    sudo firewall-cmd --list-all
    ```
1. If there is no rule in place which allows you to connect to port 27017, you have to add an exception for the *master-node* of the clusters you want work with:
    ```bash
    sudo firewall-cmd --zone=work --add-rich-rule 'rule family=ipv4 source address=<IP-address-of-cluster-master-node> port port=27017 protocol=tcp accept' --permanent`        
    ```
    Some LNM specific IP addresses are:
    - Schmarrn: 129.187.58.24
    - Bruteforce (master node global IP): 129.187.58.13
    - Deep (master node global IP): 129.187.58.20
    
1. To apply the changes run: 
    ```
    sudo firewall-cmd --reload
    ```
1. To enable passwordless access onto the localhost itself you need to copy the `ssh-key` of the localhost to the
   `authorized_keys` files by typing:
    ```
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    ```
  
[↑ Contents](#contents)
## Building the documentation
QUEENS uses [SPHINX](https://www.sphinx-doc.org/en/master/) to automatically build a html-documentation from docstring.
To build it, navigate into the doc folder with:   
```bash
cd <queens-base_dir>/doc
```  
and run:
```bash
make html
```
After adding new modules or classes to QUEENS, one needs to rebuild the autodoc index by running:  
```bash
sphinx-apidoc -o doc/source pqueens -f -M
```
before the make command.

[↑ Contents](#contents)
## Run the test suite
QUEENS has a couple of unit and regression test.
The testing strategy is more closely described in [TESTING.md](TESTING.md) 
To run the test suite type:
```bash  
pytest pqueens/tests -W ignore::DeprecationWarning
```
For more info see the [pytest documentation](https://docs.pytest.org/en/latest/warnings.html).

[↑ Contents](#contents)
## Additional information
Some further information that might be interesting when working with *QUEENS* please have a look at our wiki or check 
out the subsequent points.
### Make your machine a GitLab testing machine
To get started with Gitlab checkout the help section [here](https://gitlab.lrz.de/help/ci/README.md).
CI with with gitlab requires the setup of so called runners that build and test the code.
To turn a machine into a runner, you have to install some software as described
[here](https://docs.gitlab.com/runner/install/linux-repository.html)
The steps are as follows:
1. For RHEL/CentOS/Fedora run
    ```bash
    curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh | sudo bash
    ```
1. To install run:  
    ```bash
    sudo yum install gitlab-runner
    ```
1. Next, you have to register the runner with your gitlab repo as described
[here](https://docs.gitlab.com/runner/register/index.html).

[↑ Contents](#contents)
### Anaconda tips and tricks
1. Create new anaconda environment  
`conda create -n <name_of_new_environment> python=3.8`
2. List all packages linked into an anaconda environment  
`conda list -n <your_environment_name`
3. Activate environment  
`source activate <your_environment_name>`
