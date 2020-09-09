# Contributing to QUEENS
This file contains notes for potential contributors to QUEENS, as well as some notes that may be helpful for maintenance.

## Project scope
We do welcome contributions to QUEENS. However, the project is deliberately of limited scope, to try to ensure a high quality codebase.

## Code Style
 - Python code should follow roughly the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. We allow exceptions for, e.g., capital (single-letter) variable names to correspond to the notation of a paper (matrices, vectors, etc.). To help with this, we suggest using a linter plugin for your editor.
 - Practise good code as far as is reasonable. Simpler is usually better. Avoid using overly complicated language features. Reading the existing QUEENS code should give a good idea of the expected style.
 - These videos might help as well: [one](https://www.youtube.com/watch?v=OSGv2VnC0go) and [two](https://www.youtube.com/watch?v=wf-BqAjZb8M)
 - Some further resources can be found [here](http://neckbeardrepublic.com/screencasts/)
 - The book *Fluent Python* is also highly recommended.

## Pull requests and the master branch
All code that is destined for the master branch of QUEENS should initially be developed in a separate feature  branch.
Only a small number of people can merge feature branches onto the master branch (currently Jonas Biehler).

## Tests and continuous integration
QUEENS is covered by the testing suite. We expect changes to code to pass these tests, and for new code to be covered by new tests. Currently, coverage is reported by codecov.

## Use of third party open-source software 
Please do not use any third party code or library before having read 
the [wiki page about our open-source license policy](https://gitlab.lrz.de/jbi/queens/-/wikis/third-party-software). 

## Documentation
QUEENS documentation is not comprehensive, but covers enough to get users started. We expect that new features have documentation that can help other get up to speed.
