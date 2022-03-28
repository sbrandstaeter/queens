# Contributing to *QUEENS*
Thank you very much for your willingness to contribute to *QUEENS*! We strongly believe in the synergy effect of developing and using *QUEENS* together as a community.
This file contains guidelines for contributors, developers and maintainers in QUEENS.


We do welcome contributions to QUEENS. However, the project is deliberately of limited scope, to try to ensure a high quality codebase.
### Content
- [Software license and project mindset](#software-license-and-project-mindset)
- [Project organization](#project-organization)
- [The *QUEENS* development workflow](#the-queens-development-workflow)
  - [Gitlab issues](#gitlab-issues)
  - [Test your code](#test-your-code)
  - [Preparing a commit](#preparing-a-commit)
  - [Staying up-to-date with the master branch](#staying-up-to-date-with-the-master-branch)
  - [Pushing your development branch to the remote](#pushing-your-development-branch-to-the-remote)
  - [Merging another branch into master (merge requests, (MR))](#merging-another-branch-into-master-merge-requests-mr)
  - [Keep things clean](#keep-things-clean)
  - [Actively watch the pipeline / continuous integration (CI)](#actively-watch-the-pipeline--continuous-integration-ci)
  - [Working with epics and milestones](#working-with-epics-and-milestones)
- [Working with forks](#working-with-forks)
- [Coding style in *QUEENS*](#coding-style-in-queens)
- [Reading and writing documentation](#reading-and-writing-documentation)
- [Website](#website)

## Software license and project mindset
TBD
Please do not use any third party code or library before having read the [wiki page about our open-source license policy](https://gitlab.lrz.de/queens_community/queens/-/wikis/third-party-software).


## Project organization
*QUEENS* is an open source project that is currently hosted by the [Institute for Computational Mechanics](https://www.epc.ed.tum.de/en/lnm/home-en/) at the [Technical University of Munich](https://www.tum.de/en/). Within the Gitlab *QUEENS* group (https://gitlab.lrz.de/queens_community) we have several subgroups that we use for internal organization purposes, as well as the actual queens-project (https://gitlab.lrz.de/queens_community/queens).

The most important subgroups are:
-  [**QUEENS_maintainers**](https://gitlab.lrz.de/queens_community/QUEENS_maintainers): All members of this group have also Gitlab-maintainer rights. This means they are the only ones who can approve and merge a merge request or organize epics and further internal planning in *QUEENS*.
For now, this group consists of the active core developer team at LNM. Everybody can request a membership of this group by contacting a current maintainer. The current maintainers will then decide if a membership is justified.

- [**QUEENS_contributers**](https://gitlab.lrz.de/queens_community/queens_contributers): This group contains all active developers of *QUEENS* that want to contribute and shape the project. Members will be listed on our website and software publications (given that respective contributions are committed and merged in our master branch). Please note, that all maintainers are automatically also in this group. Furthermore, this group has access to the our [developer meetings project](https://gitlab.lrz.de/queens_community/developer_meetings), which take place in a bi-weekly turns. The protocols of previous meetings are also published in this project. Everybody can request access to this group by contacting one of the maintainers. The group membership can also be limited on a shorter time period. We will open an onboarding and farewell issue to welcome and say goodbye to members.

-  [**students**](https://gitlab.lrz.de/queens_community/students): This group contains all current students that work within the *QUEENS* LNM fork along with all other contributors and maintainers. In this way we can keep the overview for the current student projects in QUEENS and also give this group special access rights to the necessary projects.

All groups can access the queens-project (actual code) but internally it helps our organization and overview to assign members to the appropriate group. Gitlab furthermore distinguishes the [permissions or roles](https://docs.gitlab.com/ee/user/permissions.html) *owner*, *maintainer*, *developer* and *guest*, which give certain permissions within Gitlab, as the naming suggests. *Owner* of the *QUEENS* group/project is Prof. Wall along with a PostDoc and one representative PhD student. All members of the **QUEENS_maintainer** group have the Gitlab *maintainer* permissions. All other members of *QUEENS* have *developer* permissions. *Guest* permissions are not used in this project.


----
## The *QUEENS* development workflow
In the following please find some points that illustrate the development workflow in *QUEENS*. For the overall process we would like to point the reader to the well-defined development workflow of [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow#:~:text=The%20overall%20flow%20of%20Gitflow,merged%20into%20the%20develop%20branch). Below please find some additional remarks that will help to optimize your development workflow:
### Gitlab issues
Issues are generally used to remind or inform yourself or others about certain things in the software. Classically they are used to report bugs in the code or start a feature request or plan different upcoming tasks in *QUEENS*.

Navigate to our Gitlab issue page in [list-view](https://gitlab.lrz.de/queens_community/queens/-/issues) or [board-view](https://gitlab.lrz.de/queens_community/queens/-/boards) to get a first overview and to open new issues.

When you select our issue template, you can find different fields to tag other developers, give a short description of your issue and give the tasks a deadline or assign yourself to the issue. For bug reports or similar issues this might not be possible or beneficial right away.

We use the [Gitlab labeling system](https://docs.gitlab.com/ee/user/project/labels.html) to organize our issues in a useful manner. Please visit the [labeling page](https://gitlab.lrz.de/queens_community/queens/-/labels) to see our available labels. It is compulsory that every issue has a label out of each of the following categories:
- status label
- topic label
- type label

More labels can of course be assigned if they contribute to categorizing the issue. Please note that the scoped labels (the ones that are only half colored) can only be selected once.

Some further (optional) helpful labels might be:
- The `quick-fix` label: For issues that can be easily resolved under one hour
- The `knowledge-level:<expert/intermediate/beginner>` label which indicates the required knowledge level to solve the issue

Before you open a new issue please check within the existing issues if your bug or question were already reported. Filter for the respective labels to simplify your search.

### Test your code
At latest before you start a new merge request but better already beforehand, make yourself familiar with our [testing guidelines](https://gitlab.lrz.de/queens_community/queens/-/blob/master/pqueens/tests/README.md). It is an essential part of code development to check if our changes or additional code breaks existing untittests or integration tests. By running tests locally on a regular basis, you can make sure that problems are detected in an early stage. Please note, that your code must lead to a passing pipeline (all tests must pass) so that it can be reviewed and merged into the master branch. Furthermore, we require that new code must be tested by unittests and potentially integration tests before it can be merged. Writing tests and checking the existing tests on a regular basis helps you to write better and easier to maintain code.

### Preparing a commit
To keep track of changes on your local development branch we would like to encourage you to often commit the current state of your branch with a [git commit](https://www.git-tower.com/learn/git/commands/git-commit). Please use a meaningful commit message. After you configured our git-hooks as [described in the README](https://gitlab.lrz.de/queens_community/queens/-/blob/master/README.md) the [prepare-commit-msg](https://github.com/commitizen/cz-cli) hook will also help to formulate a commit message. Furthermore, our pre-commit hooks will already format the code to be compliant with our coding style or let you know at which places you need to adjust your code. This way, your code should always be compliant to our `code-style check` in your testing pipelines and you won't encounter big surprises in a merge request.

Our commit-msg and prepare-commit-msg hooks follow the [conventional commits standard](https://www.conventionalcommits.org/en/v1.0.0/).

### Staying up-to-date with the master branch
Even when you are currently only working on your personal local development branch it is strongly recommended to stay in touch and up-to-date with changes of the remote master. This way you will avoid large deviations from your branch to the remote repository and later on potentially many conflicts at once. There are different ways of staying up-to-date with the master branch. You can follow the merging strategies described in [this git merge workflow](https://www.atlassian.com/git/tutorials/using-branches/git-merge) or on the [git-website](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging). Please note, that for development branches that are only local (not yet pushed to the remote), [rebasing](https://docs.gitlab.com/ee/topics/git/git_rebase.html#regular-rebase) can offer a cleaner alternative to merging the current remote master.

### Pushing your development branch to the remote
Either to share your current developments with others or to prepare a merge request into the remote master branch, you need to push your branch or the current state of your branch to the remote repository. To avoid future merge conflicts and to stay up-to-date with the current remote master branch, we strongly recommend to first conduct the points described in the section [Staying up-to-date with the master branch](#staying-up-to-date-with-the-master-branch).
Assuming you have already staged and committed all your local changes as well as integrated the current master version from the remote into your local branch, you can push your current working branch with:
```
git push --set-upstream origin <name-of-your-branch>
```

### Merging another branch into master: merge requests (MR)
We encourage you to merge your developments on a regular basis into master such that everyone can profit from the new code and your developments are also tested in our pipeline. Furthermore, it is usually less work to merge smaller pieces of work than one big merge request. **Please think also about the poor reviewers and avoid huge merge requests!**

We summarize the most important aspects of merge requests in the following bullet points:

- You can open a MR by clicking on `Merge requests` in the side bar and then `New merge request` and then by selecting the respective development branch you want to merge. The target branch is mostly `master`, unless you want to merge one of your development branches into another one.
- Please select our **merge request template** and fill it out accordingly.
- Afterwards please also assign one or more reviewers with maintainer status

Some requirements before you can merge:
- Every MR into master needs at minimum one approval in order to be merged. Approvals can only be given by a member of at least developer status in the respective repository. Please @mention developers that can approve your MR.
- Before merging, all threats have to be resolved and
- all tests/the pipeline need to pass
- the test coverage must be maintained or increased, meaning **you cannot merge untested code**.

Some further comments:
- The master branch is always protected and no direct push/merge into master is possible
- Only maintainers can merge a merge request
- A MR can be open in draft-mode which means that no direct review is needed and further work on the MR is necessary. If possible, the person who opened the draft-MR should set a time limit to indicate how long the MR stays in draft-mode
- As soon as the MR is ready for review please set a reviewer for your MR
- We should try to use the `start a review` option to collect review comments before sending them. This keeps the Email account a bit cleaner
- MRs should be reviewed as soon as possible to keep the process going. A reaction to a MR should follow within two working days. Please let the developer know in case your review might take longer due to other obligations.
- It would be nice if the MRs link to the appropriate issues, i.e., by having the issue number in the beginning of the feature branch or tagging the issues explicitly.

> **Note**: Be careful when using `git rebase -i` on commits that have already been pushed to the remote. However, it is possible to squash all your commits (the orginal commits already pushed to the remote and newer commits from the review process) before merging so that the history stays clean. Gitlab still keeps the different versions since merge request versions are based on push not on commit (for more info see [here](https://docs.gitlab.com/ee/user/project/merge_requests/versions.html)).

Please note, that the above points mostly discussed the procedure of merging a development branch into the remote master branch. Please make sure that your feature branch contains the latest updates from master by following [Staying up-to-date with the master branch](#staying-up-to-date-with-the-master-branch).
### Keep things clean
To keep your local repository clean and remove already merged branches, please conduct the following steps **after** a successful merge of your development branch. We assume that you have checked out your local master branch for the following steps:
```
git fetch --prune
```
The command above will remove information about already merged *remote* branches.
To also remove your local development branch that is now merged, additionally run the command:
```
git branch -d <your-local-dev-branch>
```
Now your local development branch was deleted (the changes are now part of your local master) and you have a clean repository, again!

### Actively watch the pipeline / continuous integration (CI)
*QUEENS* is covered by the testing suite. We expect changes to code to pass these tests. The test suite is automatically triggered when you push a remote branch or when you start a merge request. Furthermore, the entire test suite runs on the master branch every night (nightly pipeline). Whereas a failing pipeline for a development branch is more an information for yourself, merge requests or the master pipeline must not lead to failing tests.
Our pipeline consists of several stages that are listed in the following.

- **Build stage**: Checks if the [Anaconda](https://www.anaconda.com/) environment can be build and if the [Singularity](https://sylabs.io/singularity/) image can be build and then also builds both of them in preparation for the upcoming stages.
- **Codechecks stage**: The stage consists of a **code analysis**, **documentation check** and a **license check**:
    - **code_analysis**: Here we check whether your code is compliant with our [coding style in *QUEENS*](#coding-style-in-queens). The pipeline or already your local commits will fail if you write code in a non-compliant style. Hence, we strongly recommend to configure our [Git-hooks as described in our README](https://gitlab.lrz.de/queens_community/queens/-/blob/master/README.md) and potentially integrate some of the code checks into your IDE. Please consult our [wiki for some helpful presets](https://gitlab.lrz.de/queens_community/queens/-/wikis/Set-up-your-Integrated-Development-Environment).
    - **documentation**: In *QUEENS* it is necessary to write documentation for your methods and classes (See [Reading and writing documentation](#reading-and-writing-documentation) for more information). In this substage, we check whether your provided docstring can be used to build a html documentation using Sphinx.
    - **license**: This check goes through all external packages and makes sure that they are compliant with our [license model](#software-license-and-project-mindset) for *QUEENS*.
- **Tests stage**: Here the actual [unittests and integration tests](https://realpython.com/python-testing/) for the code are running. We use [pytest](https://docs.pytest.org/en/6.2.x/) to manage the tests. See also our [wiki for more information](https://gitlab.lrz.de/queens_community/queens/-/wikis/pytest-framework). We trigger all tests natively on CentOs7 (unittests and integration tests) and in an Ubuntu Docker container (only unittests)
- **Pages stage**: Here we generate the coverage report for *QUEENS* and also the Sphinx documentation.
- **Cleanup stage**: Just some internal clean-up like removing the old conda environment
- **Deploy stage**: Here we deploy the pages as an actual static HTML page within gitlab

 Please note, that you can also trigger the **code analysis** as well as the **test stage** manually for your local work and it is recommended to do so on a regular basis! Please refer to the respective sections for further details.

### Working with epics and milestones
**Epics** are used for high-level project planning and are defined by the maintainers. They are used to collect issues that contribute to the respective epic and allow for an overview of the current state of the project. Contributors can on the other hand use **milestones** to group and organize their related issues. Please also see this [Gitlab discussion](https://gitlab.com/gitlab-org/gitlab/-/issues/6222) or [this video](https://youtu.be/9W4oxjdAwUs) for some elaboration on the differences.

---
## Working with forks
We follow the fork-based workflow as described in [Atlassian Bitbucket](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow). Developers can use [repository mirroring](https://docs.gitlab.com/ee/user/project/repository/mirror/index.html) to keep their fork synced with the original repository. When you are ready to send your code back to the upstream project, create a [merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html). For Source branch, choose your forked project’s branch. For Target branch, choose the original project’s branch. An exemplary fork workflow can also be found on [Stackoverflow](https://stackoverflow.com/questions/20956154/whats-the-workflow-to-contribute-to-an-open-source-project-using-git-pull-reque).


---
## Coding style in *QUEENS*
 Python code should follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. We check coding style in our testing pipeline in the **codecheck stage**, such that uncompliant code will lead to failing pipelines and you won't be able to merge your contributions into the master branch. The following code checks are conducted in the **codecheck stage**:

 - [vulture](https://github.com/jendrikseipp/vulture) to detect unreachable code
 - [pylint](https://pylint.org/), to check if code is compliant to PEP 8
 - [isort](https://github.com/PyCQA/isort), to check if imported modules are sorted correctly
 - [black](https://github.com/psf/black), to check if your code is formatted according to the google coding style and PEP 8
 - [pydocstyle](https://github.com/PyCQA/pydocstyle), to check if docstrings exists and if they are compliant to the [PEP 257](https://www.python.org/dev/peps/pep-0257/) guidelines for Python docstrings


 To help you with writing PEP 8 compliant code, we provide [git hooks](https://www.atlassian.com/git/tutorials/git-hooks) that automatically conduct some code formatting for you and also check if you would be compliant with our pipeline checks. Additionally, to the pipeline-checks these hooks also trigger the following tools (for configuration please see our [README](https://gitlab.lrz.de/queens_community/queens/-/blob/master/README.md)):
 - [commitizen](https://github.com/commitizen-tools/commitizen), which prepares a commit message template, such that your commit messages are compliant to the
[conventional commit standard](https://www.conventionalcommits.org/en/v1.0.0/)
 - [docformatter](https://github.com/myint/docformatter) formats existing docstrings according to [PEP 257](https://www.python.org/dev/peps/pep-0257/) guidelines.

It might be advantageous to integrate some of the previous tools directly in your IDE to check your code on a continuous basis.

Practice good code as far as is reasonable. Simpler is usually better. Avoid using overly complicated language features. Reading the existing QUEENS code should give a good idea of the expected style. These videos might help as well: [one](https://www.youtube.com/watch?v=OSGv2VnC0go) and [two](https://www.youtube.com/watch?v=wf-BqAjZb8M). Some further resources can be found [here](http://neckbeardrepublic.com/screencasts/). Also please have a look at your [wiki](https://gitlab.lrz.de/queens_community/queens/-/wikis/home) where we already collected some ideas and tips for clean code.

## Reading and writing documentation
*QUEENS* uses [Sphinx](https://www.sphinx-doc.org/en/master/) to generate a HTML-documentation automatically from the docstrings in the source code. We also provide a compiled version of our documentation [here on Gitlab](https://queens_community.pages.gitlab.lrz.de/queens/docs/) (Within the [queens project](https://gitlab.lrz.de/queens_community/queens), just click on the `sphinx passed` button in the top right). We require that all methods and classes have a valid docstring that follows the Google-style. Examples for the [Google-style Python docstring](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).
Further examples can be found [here](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).

## Website
Our website is currently under construction. We will automatically list all active contributors on the project website. If you want to contribute to our website or we did not list your contribution, please contact one of the *QUEENS* maintainers.
