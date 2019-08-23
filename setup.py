import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pqueens",
    version = "0.1",
    author = "Jonas Biehler",
    author_email = "biehler@adco-engineering-gw.de",
    description = ("A package for Uncertainty Quantification and Bayesian optimization"),
    keywords = "Gaussian Processes, Uncertainty Quantification",
    packages=['pqueens',
              'pqueens.database',
              'pqueens.post_post',
              'pqueens.randomfields',
              'pqueens.drivers',
              'pqueens.interfaces',
              'pqueens.iterators',
              'pqueens.models',
              'pqueens.regression_approximations',
              'pqueens.resources',
              'pqueens.schedulers',
              'pqueens.variables',
              'pqueens.utils'],
    # for now do not add third party packages here but install them manually Using
    # anaconda beforehand (we get some weird errors otherwise)
    #install_requires=[
    #      'numpy',
    #      'scipy',
    #      'matplotlib'
    #  ],
    long_description=read('README.md'),
    setup_requires='pytest-runner',
    tests_require='pytest',
)
