# QUEENS #

This repository is contains the python version of the QUEENS framework.

#### Dependencies ####
All necessary third party libraries are listed in the requirements.txt file.
To install all of these dependencies using pip, simply run:   
`pip install -r requirements.txt`

MongoDB
Installation instruction can be found [here](https://docs.mongodb.com/master/tutorial/install-mongodb-on-os-x/?_ga=2.181134695.1149150790.1494232459-1730069423.1494232449)


#### Installation directions ####
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


#### Building the documentation ####
QUEENS uses sphinx to automatically build a html documentation from  docstring. To build it, navigate into the doc folder and type:    
`make html`  

After adding new modules or classes to QUEENS, one needs to rebuild the autodoc index by running:    
`sphinx-apidoc -o doc/source pqueens -f`  
before the make command.

### Run the test suite ###
QUEENS has a couple of unit and regression test. To run the test suite type:  
`python -m unittest discover pqueens/tests`

In order to get a detailed report showing code coverage etc., the test have to be run using the coverage tool. This is triggered by running the using:    
`coverage run -m unittest discover -s pqueens/tests`  

To view the created report, run:  
`coverage report -m`


#### Run the Bitbucket pipeline locally ####
It is possible to test Bitbucket pipelines locally on your machine using
Docker containers. The main purpose of this is to be able to try and test
the Bitbucket pipeline as well as the overall bitbucket setup locally on
your machine. This can help to fix differences between test results obtained
locally and results obtained online in the bitbucket repository. Testing
things locally in a Docker container matching the bitbucket setup repository
is often very helpful speeds up the debugging process considerably.  

Bitbuckets own documentation about hwo to test the pipelines locally
can be found
[here](https://confluence.atlassian.com/bitbucket/debug-your-pipelines-locally-with-docker-838273569.html)

In any case, the steps are fairly straightforward:
- If [Docker](https://www.docker.com/) is not already installed, install Docker on your machine by following these [instructions](https://www.docker.com/docker-mac)
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
