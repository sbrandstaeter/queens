
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
    version = "0.01",
    author = "Jonas Biehler",
    author_email = "biehler@adco-engineering-gw.de",
    description = ("A package for Uncertainty Quantification and Bayesian optimization"),
    keywords = "Gaussian Processes, Unctertainty Quantification",
    packages=['pqueens',
              'pqueens.randomfields'],
    # add required third party packages here
    #install_requires=[
    #      'numpy',
    #      'scipy',
    #      'matplotlib'
    #  ],
    long_description=read('README.md'),
)
