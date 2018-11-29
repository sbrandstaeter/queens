# QUEENS #

This repository contains the code of the QUEENS framework.

[![build status](https://gitlab.lrz.de/jbi/queens/badges/master/build.svg)](https://gitlab.lrz.de/jbi/queens/commits/master)

[![coverage report](https://gitlab.lrz.de/jbi/queens/badges/master/coverage.svg)](https://codecov.io/bitbucket/jbi35/pqueens/commit/6c9c94b0e6e8f6c2b5aad07b34e69c72a4d1edce)

## Dependencies
All necessary third party libraries are listed in the requirements.txt file.
To install all of these dependencies using pip, simply run:   
`pip install -r requirements.txt`

To be able to build the documentation, an additional package *pandoc* is needed.
Installation instructions are provided [here](http://pandoc.org/installing.html)


## Installation directions
The use a virtual environment like [Anaconda](https://www.continuum.io/downloads) is highly recommended.
After setting up Anaconda and a new, dedicated QUEENS development environment via
`conda create -n <name_of_new_environment> python=3.6`   
All required third party libraries can be simply installed by running:  
`pip install -r requirements.txt`  
Next, if Anaconda is used, QUEENS can be installed using:     
`/Applications/anaconda/envs/<your-environment-name>/bin/python setup.py develop`   
If you encounter any problems try using the --user flag.

Otherwise the command is simply:  
`python setup.py develop`

To uninstall QUEENS run:  
`python setup.py develop --uninstall`

To update Python packages in your Anaconda environment type:  
`conda update --all`

To get your anaconda environments into your Jupyter kernel run:
`conda install nb_conda_kernels`

### Setup of MongoDB
QUEENS writes results into a MongoDB database, therefore QUEENS needs to have write access to a MongoDB databases. However, MongoDB does not necessarily have to run on the same machine as QUEENS. In certain situations, it makes sense to have the database running on a different computer and connect to the database via port-forwarding.

#### Installation of MongoDB
Installation instructions if you are running OSX can be found [here](https://docs.mongodb.com/master/tutorial/install-mongodb-on-os-x/?_ga=2.181134695.1149150790.1494232459-1730069423.1494232449)  
Installation instructions for LNM workstations running Fedora 22 are as follows.
https://blog.bensoer.com/install-mongodb-3-0-on-fedora-22/   

#### Starting MongoDB
After installation, you need to start MongoDB. For LNM machines this requires that you can execute the commands:   
`systemctl start mongod`   

`systemctl stop mongod`   

`systemctl restart mongod`

It could be that you have to edit the sudoers file `/etc/sudoers.d/` together with your administrator in order get the execution rights.

On a Mac you have to run
`mongod --dbpath <path_to_your_database>`

#### LNM specific issues
In order to connect to a MongoDB instance running on one of the LNM machines, one needs to be able to connect port 27017 from a remote host.
By default,  the fire wall software `firewalld` blocks every incoming request. Hence, to enable a connections, we have add so called rules to firewalld in order to connect to the database.   

First type   
`sudo firewall-cmd --list-all`   
to see what firewall rules are already in place.

If there is no rule in place which allows you to connect to port 27017, you can add such a rule by running the following command on the machine MongoDB is running on.   
`sudo firewall-cmd --zone=work --add-rich-rule 'rule family=ipv4 source address=<adress-you-want-to-connnect-to> port port=27017 protocol=tcp accept' --permanent`   
Note that if you want to connect to the database from a cluster, you will also need to add this rule for the ip address of the clusters master node.

**Bruteforce**: In case your computing cluster is Bruteforce and MongoDB is running on your local machine, the **only** rule that should be in place is:
``sudo firewall-cmd --zone=work --add-rich-rule 'rule family=ipv4 source address=129.187.58.13 port port=27017 protocol=tcp accept' --permanent``

Also, to apply the changes run  
`sudo firewall-cmd --reload`  

### Some IP-adresses
- Cauchy: 129.187.58.39
- Bohr: 129.187.58.88
- Schmarrn: 129.187.58.24
- Bruteforce (master node global IP): 129.187.58.13
- Bruteforce (master node local IP): 10.10.0.1
- Jonas B. Laptop: 129.187.58.120

#### QUEENS and cluster jobs
QUEENS offers the possibility to perform the actual model evaluations on a HPC-cluster
such as Kaiser. Setting things up is unfortunately not really as straightforward as it could be yet. However, the following steps can be used as guidance to set up QUEENS such that QUEENS is running on one machine, the MongoDB is running on a second machine and the model evaluations are performed on a third machine, in this case Kaiser.
To avoid confusion, these three machine will be referred to as
- localhost (Local machine running QUEENS)
- db_server  (machine running MongoDB, e.g., Cauchy)
- compute machine (HPC-cluster, e.g., Kaiser)   

in the following.

##### Preparing the compute machine (e.g. Kaiser or Bruteforce)
First we have to install some software on the compute cluster. The following steps
are tried and tested on the LNM Kaiser and Bruteforce system. It is not guaranteed that the steps
will be the same on other systems

1. Install miniconda3 with python 3.6 on the cluster
2. Add public ssh-key of Cluster to LRZ GitLab repo in order to be able to clone QUEENS
3. Clone this repo
4. Install requirements and QUEENS using pip on Kaiser or Bruteforce
5. (**ONLY BRUTEFORCE:**) On Bruteforce a port-forwarding to the db-server is necessary as the computing nodes cannot reach networks outside of Bruteforce:
   1. On master node run the command `ssh -fN -g -L 27017:localhost:27017 <user>@<computer_name>.lnm.mw.tum.de`
   2. *Be careful*: Now the address of the MongoDB server in the input-file has to be changed to the global Bruteforce IP-adress with port 27017: `129.187.58.13:27017`.
   3. The command above will now forward port 27017 of the Bruteforce's master node to the local host port 27017 (e.g. your personal work station) which is the interface to the DB (running locally)
   4. Currently, the driver file used on Bruteforce contains the local IP-address of its master node (`10.10.0.1:27017`) hard-coded to enable port-forwarding from slave nodes to the external DB-server

##### Preparing the db_server
The machine running the MongoDB database does not need to be the same as either the
machine running QUEENS or the compute cluster. However, the other machines need to be
able to connect to the MongoDB via tcp in order to write data. For that to work
within the LNM network, all of the machines need to be connected to the internal network.

1. Install MongoDB. If you are using CentOS, a nice guide can be found
[here](https://www.digitalocean.com/community/tutorials/how-to-install-mongodb-on-centos-7).
2. Make sure you can connect to the database from the computer running QUEENS.
This usually entails two steps.
  1. Open the respective ports in the firewall and allowing both localhost and the
  compute machine to connect to the db_server. This can be achieved by adding
  rich-rules to firewalld using firewall-cmd as described above.
  2. Edit the MongoDB config file such that it allows connections from anywhere
  (probably not the safest option, but ok for now). If you followed the standard
  installation instructions, you should edit the config file using   
  `sudo vim /etc/mongod.conf`  
  And comment out the `bindIp` like shown   
  ```shell
  # network interfaces
   net:
   port: 27017
   # bindIp: 127.0.0.1  <- comment out this line
   ```   

##### Preparing localhost
1. Install QUEENS
2. Activate ssh-port-forwarding so that you can connect to the compute machines
 without having to enter your password. To learn more about how ssh port forwarding
 work click [here](https://chamibuddhika.wordpress.com/2012/03/21/ssh-tunnelling-explained/).
 Without explanation of all the details, here is an example. Assuming the compute machine is Kaiser
 and the db_server is cauchy, you have to run the following command on your localhost to
 set up ssh port forwarding  
 `ssh -fN -L 9001:kaiser.lnm.mw.tum.de:22 biehler@cauchy.lnm.mw.tum.de`   
 To see if port forwarding works type  
 `ssh -p 9001 biehler@localhost`  
 to connect to Kaiser.   
 Connecting via ssh to the compute machine needs to work without having to type your
 password. This can be achieved by copying your ssh key from localhost to kaiser.
 Depending on your system, you need to locate the file with the ssh keys. On my mac I can
 activate passwordless ssh to Kaiser by running the following   
`cat .ssh/id_rsa.pub | ssh biehler@kaiser 'cat >> .ssh/authorized_keys'`   
 The easiest way to disable ssh port forwarding is to run   
 `killall ssh`  
 Beware, this kills all ssh processes not just the port forwarding one.


## Building the documentation
QUEENS uses sphinx to automatically build a html documentation from  docstring.
To build it, navigate into the doc folder and type:    
`make html`  

After adding new modules or classes to QUEENS, one needs to rebuild the autodoc index by running:    
`sphinx-apidoc -o doc/source pqueens -f -M`  
before the make command.

## Run the test suite
QUEENS has a couple of unit and regression test. To run the test suite type:  
`pytest pqueens/tests`

In order to get a detailed report showing code coverage etc., the test have to be run using the coverage tool. This is triggered by running the using:    
`coverage run -m pytest pqueens/tests`  

To view the created report, run:  
`coverage report -m`

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


## Anaconda tips and tricks
1. Create new anaconda environment
`conda create -n <name_of_new_environment> python=3.6`  
2. List all packages linked into an anaconda environment
`conda list -n <your_environment_name`
3. Activate environment
`source activate <your_environment_name>
