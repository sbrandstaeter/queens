# README #

### QUEENS ###

This repository is contains the python version of the QUEENS framework.

[![Codecov private](https://img.shields.io/codecov/c/token/8eecedcc-6782-468a-b066-0e641741f210/bitbucket/codecov/example-python.svg)]()

All of the stuff below has to be considered preliminary as I am learning by doing.
#### Dependencies ####
Install those using anaconda first
- numpy
- scipy
- matplotlib
- sphinx
- pymongo

MongoDB
install
https://docs.mongodb.com/master/tutorial/install-mongodb-on-os-x/?_ga=2.181134695.1149150790.1494232459-1730069423.1494232449

#### Installation directions ####

For developers:
Install all requirements using the requirements file:
`pip install -r requirements.txt`
run `python setup.py develop` Using a virtual environment e.g. based on anaconda is highly recommended. In that case install command
should be executed as follows:
`/Applications/anaconda/envs/test/bin/python python setup.py develop`
To uninstall:
`python setup.py develop --uninstall`

For users (maybe later on):
run `run setup.py install`

Update python packages using:
`conda update --all`


#### Building the documentation ####
Navigate into the doc folder and type
`make html` to build the html documentation
After adding new modules or classes rebuild autodoc by typing first
`sphinx-apidoc -o doc/source pqueens -f`

### Run the test suite ###
`python -m unittest discover pqueens/tests`

run with coverage to get detailed test coverage report:
`coverage run -m unittest discover -s pqueens/tests`

to view report run:
`coverage report -m`


#### Installation directions for installation within Docker container ####
The main purpose of this is to be able to try and test the bitbucket pipeline
and overall bitbucket setup locally. Unfortunately there are sometimes differences
between test results obtained locally and results obtained online in the bitbucket
repository. Testing things locally in a Docker container matching the bitbucket setup
repository should help to speed up debugging.

Bitbuckets documentation on the topic can be found here:
https://confluence.atlassian.com/bitbucket/debug-your-pipelines-locally-with-docker-838273569.html

- If docker is not already installed, install docker on your machine
- Copy the code to a dedicated testing directory, e.g.,
  `/Users/jonas/work/adco/test_bibucket_pipeline_locally/pqueens`
- Run the following command:
`docker run -it --volume=/Users/jonas/work/adco/test_bibucket_pipeline_locally/pqueens:/localDebugRepo --workdir="/localDebugRepo" --memory=4g --memory-swap=4g --entrypoint=/bin/bash continuumio/anaconda3`
- You should now be in the docker container and inside the queens directory
- From here, you can now run the individual commands in your pipeline
