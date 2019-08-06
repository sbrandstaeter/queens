# QUEENS #

This repository contains the code of the QUEENS framework.

## Contents

1. [Dependencies](#dependencies)
1. [Installation directions](#installation-directions)
   1. [Setup of MongoDB](#setup-of-mongodb)
      1. [Installation of MongoDB](#installation-of-mongodb)
      1. [Starting MongoDB](#starting-mongodb)
      1. [LNM specific issues](#lnm-specific-issues)
   1. [Some IP addresses](#some-ip-addresses)
   1. [QUEENS and cluster jobs](#queens-and-cluster-jobs)
      1. [Preparing the compute machine](#preparing-the-compute-machine)
      1. [Preparing the db_server](#preparing-the-db_server)
      1. [Preparing the localhost](#preparing-the-localhost)
1. [Building the documentation](#building-the-documentation)
1. [Run the test suite](#run-the-test-suite)
1. [GitLab](#gitlab)
1. [Anaconda tips and tricks](#anaconda-tips-and-tricks)


## Installation directions
**Note**<br />
In the current version, the *QUEENS* installation has to be done twice: On the localhost (e.g. *bohr*) **and** on the compute machine (e.g. *bruteforce*).<br />
For the installation on a cluster, copy the Anaconda-executable to the desired directory and install Anaconda. <br />

1. Add your public SSH key to your GitLab user profile. For instructions on how to create and add a SSH key look at the [GitLab documentation](https://gitlab.lrz.de/help/ssh/README).
1. You can then clone this repository to your local machine with  
`git clone git@gitlab.lrz.de:jbi/queens.git <target-directory>`  
1. [Install](http://docs.anaconda.com/anaconda/install/linux/) the latest version of [Anaconda](https://www.anaconda.com/) with Python 3.x.
 *Anaconda* is an open-source Python distribution containing many popular data science Python packages. It includes a powerful package and virtual environment manager program called *conda*. 
1. After setting up Anaconda on your machine, create a new, dedicated QUEENS development environment via  
`conda create -n <name_of_new_environment> python=3.7`
1. You need to activate the newly created environment via  
`conda activate <name_of_new_environment>`
1. All required third party libraries can then be simply installed in the environment by running:  
`pip install -r <your-path-to-QUEENS>/requirements.txt`
1. Now, QUEENS can be installed in the environment using:  
`python <your-path-to-QUEENS>/setup.py develop`  
If you encounter any problems try using the --user flag.

**Documentation:**<br />
To be able to build the documentation, an additional package *pandoc* is needed.
Installation instructions are provided [here](http://pandoc.org/installing.html)

**Update packages:**<br />
To update Python packages in your conda environment, type:  
`conda update --all`

**Use conda environments in Jupyter notebooks:**<br />
To get your conda environments into your Jupyter kernel, run:  
`conda install nb_conda_kernels`

**Uninstall:**<br />
To uninstall *QUEENS* from your conda environment, run  
`python setup.py develop --uninstall`  
within the activated environment.

[↑ Contents](#contents)

### Setup of MongoDB
QUEENS writes results into a MongoDB database, therefore QUEENS needs to have write access to a MongoDB databases. However, MongoDB does not necessarily have to run on the same machine as QUEENS. In certain situations, it makes sense to have the database running on a different computer and connect to the database via port-forwarding.

[↑ Contents](#contents)

####  Installation of MongoDB

We strongly recommend the use of MongoDB version 3.4 as this is proven to work in the current framework.

- Installation instructions if you are running **macOS** can be found [here](https://docs.mongodb.com/master/tutorial/install-mongodb-on-os-x/?_ga=2.181134695.1149150790.1494232459-1730069423.1494232449)
- Installation instructions for LNM workstations running **CentOS 7** can be found [here](https://www.digitalocean.com/community/tutorials/how-to-install-mongodb-on-centos-7).  The Repository file for other MongoDB versions can be found in the official [MongoDB documentation](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-red-hat/).
- Installation instructions for LNM workstations running **Fedora 22** can be found [here](https://blog.bensoer.com/install-mongodb-3-0-on-fedora-22/)

[↑ Contents](#contents)

#### Starting MongoDB
After installation, you need to start MongoDB.
1. For LNM machines this requires that you can execute the commands:  
`sudo systemctl start mongod`  
`sudo systemctl stop mongod`  
`sudo systemctl restart mongod`  
It could be that you have to edit the sudoers file `/etc/sudoers.d/` together with your administrator in order get the execution rights.

1. On a Mac you have to run:  
`mongod --dbpath <path_to_your_database>`

[↑ Contents](#contents)

#### LNM specific issues
In order to connect to a MongoDB instance running on one of the LNM machines, one needs to be able to connect port 27017 from a remote host.
By default,  the fire wall software `firewalld` blocks every incoming request. Hence, to enable a connections, we have add so called rules to firewalld in order to connect to the database.

First type  
`sudo firewall-cmd --list-all`   
to see what firewall rules are already in place.

1. If there is no rule in place which allows you to connect to port 27017, you can add one by running the following command on the machine MongoDB is running on:  
`sudo firewall-cmd --zone=work --add-rich-rule 'rule family=ipv4 source address=<ip-address-you-want-to-connnect-to> port port=27017 protocol=tcp accept' --permanent`  
You may find some LNM specific IP addresses [here](#some-ip-addresses).
Note that if you want to connect to the database from a cluster, you will also need to add this rule for the IP address of the clusters master node.  
**Bruteforce**: In case your computing cluster is Bruteforce and MongoDB is running on your local machine, the **only** rule that should be in place is:  
`sudo firewall-cmd --zone=work --add-rich-rule 'rule family=ipv4 source address=129.187.58.13 port port=27017 protocol=tcp accept' --permanent`  
**_Important_:** The port number (in the current example 27017) might already be taken by another user as the forwarding is managed over Bruteforce's master node that can be accessed by the whole group. If the rule was already set in place by another coworker, the port will be taken and the next person has to choose a port number which is free. The current work-around is to randomly pick a port number between 1025 and 65535 and try another number if a port is already taken. We should automate that in the future!
1. Finally, to apply the changes run  
`sudo firewall-cmd --reload`

[↑ Contents](#contents)

### Some IP addresses
- Cauchy: 129.187.58.39
- Bohr: 129.187.58.88
- Schmarrn: 129.187.58.24
- Bruteforce (master node global IP): 129.187.58.13
- Bruteforce (master node local IP): 10.10.0.1
- Jonas B. Laptop: 129.187.58.120

[↑ Contents](#contents)

### QUEENS and cluster jobs
QUEENS offers the possibility to perform the actual model evaluations on a HPC-cluster
such as Kaiser. Setting things up is unfortunately not really as straightforward as it could be yet. However, the following steps can be used as guidance to set up QUEENS such that QUEENS is running on one machine, the MongoDB is running on a second machine and the model evaluations are performed on a third machine, in this case Kaiser.
To avoid confusion, these three machine will be referred to as
- localhost (Local machine running QUEENS)
- db_server  (machine running MongoDB, e.g., Cauchy)
- compute machine (HPC-cluster, e.g., Kaiser or Bruteforce)

in the following.

[↑ Contents](#contents)

#### Preparing the compute machine
First we have to install some software on the compute machine. The following steps
are tried and tested on the LNM Kaiser and Bruteforce system. It is not guaranteed that the steps
will be the same on other systems.

1. Follow the [installation directions](#installation-directions) as described above to setup QUEENS on the compute machine. It might suffice to install [miniconda3](https://conda.io/miniconda.html) instead of Anaconda on the compute machine.
**Note:** Currently it is necessary to have duplicate (identical) installations of QUEENS on the localhost and on all compute machines. This should be a temporary solution.
1. (**ONLY BRUTEFORCE:**) On Bruteforce a port-forwarding to the db-server is necessary as the computing nodes cannot reach networks outside of Bruteforce:
   1. On master node run the command  
   `ssh -fN -g -L 27017:localhost:27017 <user>@<computer_name>.lnm.mw.tum.de`
   1. *Be careful*: Now the address of the MongoDB server in the input-file has to be changed to the global Bruteforce IP address with port 27017: `129.187.58.13:27017`.
   1. The command above will now forward port 27017 of the Bruteforce's master node to the local host port 27017 (e.g. your personal work station) which is the interface to the DB (running locally)
   1. Currently, the driver file used on Bruteforce contains the local IP address of its master node (`10.10.0.1:27017`) hard-coded to enable port-forwarding from slave nodes to the external DB-server

[↑ Contents](#contents)

#### Preparing the db_server
The machine running the MongoDB database does not need to be the same as either the
machine running QUEENS or the compute cluster. However, the other machines need to be
able to connect to the MongoDB via tcp in order to write data. For that to work
within the LNM network, all of the machines need to be connected to the internal network.

1. [Install MongoDB](#installation-of-mongodb).
1. Make sure you can connect to the database from the computer running QUEENS.
This usually entails two steps:
   1. Open the respective ports in the firewall and allowing both localhost and the compute machine to connect to the db_server. This can be achieved by adding rich-rules to firewalld using firewall-cmd as described above.
   1. Edit the MongoDB config file such that it allows connections from anywhere (probably not the safest option, but ok for now). If you followed the standard installation instructions, you should edit the config file using  
   `sudo vim /etc/mongod.conf`  
   And comment out the `bindIp` like shown
      ```shell
      # network interfaces
      net:
      port: 27017
      # bindIp: 127.0.0.1  <- comment out this line
      ```

[↑ Contents](#contents)

#### Preparing the localhost
1. [Install QUEENS](#installation-directions).
1. Activate ssh-port-forwarding so that you can connect to the compute machines
 without having to enter your password. To learn more about how ssh port forwarding
 works click [here](https://chamibuddhika.wordpress.com/2012/03/21/ssh-tunnelling-explained/).
 Without explanation of all the details, here is an example. Assuming the compute machine is Kaiser and the db_server is cauchy, you have to run the following command on your localhost to set up ssh port forwarding  
 `ssh -fN -L 9001:kaiser.lnm.mw.tum.de:22 <user>@cauchy.lnm.mw.tum.de`  
 To see if port forwarding works type  
 `ssh -p 9001 <user>@localhost`  
 to connect to Kaiser.  
1. Connecting via ssh to the compute machine needs to work without having to type your password. This can be achieved by copying your ssh key from localhost to kaiser.
 Depending on your system, you need to locate the file with the ssh keys. On my mac I can
 activate passwordless ssh to Kaiser by running the following  
`cat .ssh/id_rsa.pub | ssh biehler@kaiser 'cat >> .ssh/authorized_keys'`  
1. The easiest way to disable ssh port forwarding is to run  
 `killall ssh`  
 Beware, this kills all ssh processes not just the port forwarding one.

[↑ Contents](#contents)

## Building the documentation
QUEENS uses sphinx to automatically build a html documentation from  docstring.
To build it, navigate into the doc folder and type:  
`make html`  

After adding new modules or classes to QUEENS, one needs to rebuild the autodoc index by running:  
`sphinx-apidoc -o doc/source pqueens -f -M`  
before the make command.

[↑ Contents](#contents)

## Run the test suite
QUEENS has a couple of unit and regression test. To run the test suite type:  
`pytest pqueens/tests`  

The above potentially gives many deprecation warnings of external packages (like tensorflow).
If you are sure you did not cause a deprecation warning yourself, you can ignore them by running  
`pytest pqueens/tests -W ignore::DeprecationWarning`  
For more info see the [pytest documentation](https://docs.pytest.org/en/latest/warnings.html).

In order to get a detailed report showing code coverage etc., the test have to be run using the coverage tool. This is triggered by running the using:  
`coverage run -m pytest pqueens/tests`  

To view the created report, run:  
`coverage report -m`  

[↑ Contents](#contents)

## GitLab
To get started with Gitlab checkout the help section [here](https://gitlab.lrz.de/help/ci/README.md)

CI with with gitlab requires the setup of so called runners that build and test the code.
To turn a machine into a runner, you have to install some software as described
[here](https://docs.gitlab.com/runner/install/linux-repository.html)
As of 02/2018 the steps are as follows:
1. For RHEL/CentOS/Fedora run  
`curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh | sudo bash`
2. To install run:  
`sudo yum install gitlab-runner`


Next, you have to register the runner with your gitlab repo as described
[here](https://docs.gitlab.com/runner/register/index.html)

[↑ Contents](#contents)

## Anaconda tips and tricks
1. Create new anaconda environment  
`conda create -n <name_of_new_environment> python=3.7`
2. List all packages linked into an anaconda environment  
`conda list -n <your_environment_name`
3. Activate environment  
`source activate <your_environment_name>`

[↑ Contents](#contents)

