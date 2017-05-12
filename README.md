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

run `python setup.py develop` Using a virtual environment e.g. based on anaconda is highly recommended
To uninstall:
`python setup.py develop --uninstall`

For users (maybe later on):
run `run setup.py install`


#### Building the documentation ####
Navigate into the doc folder and type
`make html` to build the html documentation
After adding new modules or classes rebuild autodoc by typing first
`sphinx-apidoc -o doc/source pqueens -f`

### Run the test suite ###
`python -m unittest discover pqueens/tests`

run with coverage to get detailed test coverage report

`coverage run -m unittest discover -s pqueens/tests`

to view report run

`coverage report -m`
