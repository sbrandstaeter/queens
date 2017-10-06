# QUEENS #

This repository is contains the python version of the QUEENS framework.

[![build status](https://gitlab.lrz.de/jbi/queens/badges/master/build.svg)](https://gitlab.lrz.de/jbi/queens/commits/master)

[![coverage report](https://gitlab.lrz.de/jbi/queens/badges/master/coverage.svg)](https://gitlab.lrz.de/jbi/queens/commits/master)

## Dependencies
All necessary third party libraries are listed in the requirements.txt file.
To install all of these dependencies using pip, simply run:   
`pip install -r requirements.txt`


## Installation directions
The use a virtual environment like [Anaconda](https://www.continuum.io/downloads) is highly recommended.
After setting up Anaconda and a new, dedicated QUEENS development environment, all required third party libraries can be simply installed by running:  
`pip install -r requirements.txt`  
Next, if Anaconda is used, QUEENS can be installed using:     
`/Applications/anaconda/envs/<your-environment-name>/bin/python setup.py develop`  
Otherwise the command is simply:  
`python setup.py develop`

To uninstall QUEENS run:  
`python setup.py develop --uninstall`

To update Python packages in your Anaconda environment type:  
`conda update --all`

### Setup of MongoDB
QUEENS writes results into a MongoDB database, therefore QUEENS needs to have write access to a MongoDB databases. However, MongoDB does not necessarily have to run on the same machine as QUEENS. In certain situations, it makes sense to have the database running on a different computer and connect to the database via port-forwarding.

#### Installation of MongoDB
Installation instructions if you are running OSX can be found [here](https://docs.mongodb.com/master/tutorial/install-mongodb-on-os-x/?_ga=2.181134695.1149150790.1494232459-1730069423.1494232449)  
Installation instructions for LNM workstations running Fedora 22 are as follows.
https://blog.bensoer.com/install-mongodb-3-0-on-fedora-22/   
After installation, you need to start MongoDB. For LNM machines this requires that you can execute the commands:   
`systemctl start mongod`   

`systemctl stop mongod`   

`systemctl restart`

It could be that you have to edit the sudoers file `/etc/sudoers.d/` together with your administrator in order get the execution rights.


#### LNM specific issues
In order to connect to a MongoDB instance running on one of the LNM machines, one needs to be able to connect port 27017 from a remote host.
By default,  the fire wall software `firewalld` blocks every incoming request. Hence, to enable a connections, we have add so called rules to firewalld in order to connect to the database.   

First type   
`sudo firewalld —list-all`   
to see what firewall rules are already in place.
If there is no rule in place which allows you to connect to port 27017, you can add such a rule by running the following command on the machine MongoDD is running on.   
`sudo firewalld —zone=work —add-rich-rule ‘rule family=ipv4 source address=<ip-adress-you-want-to-connect-from> port port=27017 protocol=tcp accept’ —permanent`   
Note that if you want to connect to the database from a cluster, you will also need to add this rule to the clusters master node.


## Building the documentation
QUEENS uses sphinx to automatically build a html documentation from  docstring. To build it, navigate into the doc folder and type:    
`make html`  

After adding new modules or classes to QUEENS, one needs to rebuild the autodoc index by running:    
`sphinx-apidoc -o doc/source pqueens -f`  
before the make command.

## Run the test suite
QUEENS has a couple of unit and regression test. To run the test suite type:  
`python -m unittest discover pqueens/tests`

In order to get a detailed report showing code coverage etc., the test have to be run using the coverage tool. This is triggered by running the using:    
`coverage run -m unittest discover -s pqueens/tests`  

To view the created report, run:  
`coverage report -m`


## Run the Bitbucket pipeline locally
It is possible to test Bitbucket pipelines locally on your machine using
Docker containers. The main purpose of this is to be able to try and test
the Bitbucket pipeline as well as the overall bitbucket setup locally on
your machine. This can help to fix differences between test results obtained
locally and results obtained online in the bitbucket repository. Testing
things locally in a Docker container matching the bitbucket setup repository
is often very helpful speeds up the debugging process considerably.  

Bitbuckets own documentation about how to test the pipelines locally
can be found
[here](https://confluence.atlassian.com/bitbucket/debug-your-pipelines-locally-with-docker-838273569.html)

In any case, the steps are fairly straightforward:  

- If [Docker](https://www.docker.com/) is not already installed, install Docker on your machine
   by following these [instructions](https://www.docker.com/docker-mac)
- Copy the QUEENS code to a dedicated testing directory, e.g.,
  `/Users/jonas/work/adco/test_bibucket_pipeline_locally/pqueens`
- Run the following command to launch a Docker container with a Debian based Anaconda image and the source code from testing directory mounted as local folder in the Docker container:  
```shell
docker run -it --volume=/Users/jonas/work/adco/test_bibucket_pipeline_locally/pqueens:/localDebugRepo
--workdir="/localDebugRepo" --memory=4g --memory-swap=4g  
--entrypoint=/bin/bash continuumio/anaconda3
```
- You should now be in the docker container and inside the queens directory
- Setup QUEENS following the instructions above
- From here, you can now run the individual commands in your pipeline.


## GitLab
To get started with Gitlab checkout the help section [here](https://gitlab.lrz.de/help/ci/README.md)

CI with with gitlab requires the setup of so called runners that build and test the code.
To turn a machine into a runner, you have to install some software as described
[here](https://docs.gitlab.com/runner/install/linux-repository.html)

Next, you have to register the runner with your gitlab repo as described
[here](https://docs.gitlab.com/runner/register/index.html)
