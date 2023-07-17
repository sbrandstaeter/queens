# QUEENS
This repository contains the code of *QUEENS*, a general purpose framework for Uncertainty Quantification,
Physics-Informed Machine Learning, Bayesian Optimization, Inverse Problems and Simulation Analytics on distributed computer
systems.

## Contents
- [QUEENS](#queens)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
    - [Mamba (or conda)](#mamba-or-conda)
      - [Installation](#installation)
      - [Usage](#usage)
    - [Optional: MongoDB](#optional-mongodb)
      - [Installation](#installation-1)
    - [Optional: Singularity](#optional-singularity)
      - [Installation](#installation-2)
      - [Building a QUEENS singularity image](#building-a-queens-singularity-image)
    - [Git](#git)
  - [QUEENS Installation](#queens-installation)
  - [Start a *QUEENS* run](#start-a-queens-run)
  - [Further Topics](#further-topics)
    - [Remote Computing](#remote-computing)
  - [Documentation](#documentation)
  - [Run Test Suite](#run-test-suite)


## Prerequisites
There are various prerequisites for QUEENS such as (an appropriately configured) Git, Anaconda, and MongoDB.

[↑ Contents](#contents)

### Mamba (or conda)
QUEENS relies on [mamba](https://github.com/mamba-org/mamba) or [conda](https://docs.conda.io/en/latest/)
as package management system and environment management system.
[Mamba](https://github.com/mamba-org/mamba) is a more performant reimplementation of conda.
Either of them will work, but we strongly recommend to use **mamba** due to the performance gain.

#### Installation
There are multiple ways of installing both mamba and conda.
However, we recommend the following:
- **mamba** (recommended): install [mambaforge](https://github.com/conda-forge/miniforge#mambaforge). For detailed instructions see also [here](https://mamba.readthedocs.io/en/latest/installation.html#installation).
- conda: install [miniforge](https://github.com/conda-forge/miniforge#miniforge3) or [miniconda](https://docs.conda.io/en/latest/miniconda.html#miniconda)

#### Usage
For instructions on how to use mamba or conda, e.g. on how to activate an environment, please refer to the official guides:
- [mamba quickstart instructions](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#quickstart)
- [getting started with conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html#managing-environments)

[↑ Contents](#contents)

### Optional: MongoDB
QUEENS uses a [MongoDB](https://www.mongodb.com/) database (Community Edition) for the handling of certain data.
Therefore, QUEENS requires certain write and access rights.
MongoDB does not necessarily have to run on the same machine as QUEENS, although this is the default case.
In certain situations, though, it might make sense to have the database running on a different computer and connect to the database via port-forwarding.

#### Installation
For installation on various OS, please follow the official [installation instructions](https://www.mongodb.com/docs/manual/administration/install-community/).

>Note: If you are experiencing SELinux warnings follow the solution [here](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-red-hat/#optional-suppress-ftdc-warnings).

[↑ Contents](#contents)

### Optional: Singularity

[Singularity](https://docs.sylabs.io/guides/3.11/user-guide/) containers are well suited for being used with QUEENS.
If you are interested in using Singularity containers for your computations,  please make sure
that `Singularity` is installed on your local machine.

#### Installation
1. Please follow the official [installation instructions](https://docs.sylabs.io/guides/3.11/user-guide/quick_start.html#quick-installation-steps).

   **Tip**: we recommend to follow the "download SingularityCE from a release" option.

1. QUEENS uses the fakeroot option of singularity.
   For more information please refer to the singularity documentation for [users](https://sylabs.io/guides/3.5/user-guide/fakeroot.html) and [admins](https://sylabs.io/guides/3.5/admin-guide/user_namespace.html#config-fakeroot).

   To enable the fakeroot option for your user (on a linux machine) execute the following command once on your workstation:
   ```
   sudo singularity config fakeroot --add $USER
   ```

   To check that the command was successful, you can
   1. show the generated `/etc/subuid`
       ```bash
       cat /etc/subuid
       ```
      which should return something like `1000:4294836224:65536`.
   1. show the generated `/etc/subgid`
       ```bash
       cat /etc/subgid
       ```
      which should return something like `1000:4294836224:65536`.

#### Building a QUEENS singularity image
Assuming singularity  and QUEENS are already installed, an image for a QUEENS run can be build
(in the environment where QUEENS is installed) using the QUEENS CLI with the command:
```
queens-build-singularity
```
[↑ Contents](#contents)

### Git
> **Note** this can be skipped if you have already configured Git for other projects

A Git version >= 2.9 is required. <!-- We need at least this version to be able to configure the path to the git-hooks as outlined below. -->
Consult the official [Git documentation](www.git-scm.org) to obtain a more recent Git installation if necessary.

1. Set your username to your full name, i.e., first name followed by last name,
and your email address to your institute email address with the following commands:

    ```bash
    git config --global user.name "<Firstname> <Lastname>"
    git config --global user.email <instituteEmailAddress>
    ```

1. Set a default text editor that will be used whenever you need to write a message in Git. To set `vim` as your default text editor, type:

    ```bash
    git config --global core.editor vim
    ```

    > **Note:** Another popular choice is `kwrite`.

1. Configure [`git blame`](https://git-scm.com/docs/git-blame) to automatically ignore uninteresting revisions by running:

    ```bash
    git config blame.ignoreRevsFile .git-blame-ignore-revs
    ```

1. Configure our `git-hooks`. We use the the [pre-commit](https://pre-commit.com/) package to manage all our git hooks, automatically. Please note, that we also have guidelines in place to structure our commit messages. Here we use the [convential commits guidlines](https://www.conventionalcommits.org/en/v1.0.0/) which are enforced by our commit-msg hook (managed by [commitizen](https://github.com/commitizen-tools/commitizen)). After [cloning the repository](#clone-the-repository) into the directory `<someBaseDir>/<sourceDir>` and with an [activated QUEENS conda environment](#anaconda) run:

    ```bash
    pre-commit install --install-hooks --overwrite
    pre-commit install --hook-type commit-msg
    ```

1. In case you are using GitLab for the first time on your machine: Add your public SSH key to your GitLab
user profile under the section `User settings - SSH keys`
    1. Check for an existing SSH key pair using:
        ```bash
        cat ~/.ssh/id_rsa.pub
        ```
    1. In case of an empty response, a new key pair can be generated using:
        ```bash
        ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
        ```
    1. For further instruction on how to configure your SSH key correctly please have a look at the
     [GitLab documentation](https://docs.gitlab.com/ee/user/ssh.html#add-an-ssh-key-to-your-gitlab-account).

[↑ Contents](#contents)



## QUEENS Installation
To install *QUEENS* follow the instructions below.
It needs to be installed once on the local machine, allowing for both local and remote computing.

> **Note**: replace mamba with conda in the instructions below if you are using plain conda.
>
1. After having [Git](#configuration-of-git) availabe and appropriately configured, this repository might be cloned to your local machine:
    ```
    git clone git@gitlab.lrz.de:queens_community/queens.git <target-directory>
    ```
1. Assuming that [mamba](#mamba-or-conda) is installed on your local machine, create a QUEENS environment:
    ```
    cd <your-path-to-QUEENS>
    mamba env create
    ```

    For a custom environment name:
    ```
    mamba env create --name <your-custom-env-name>
    ```
1. Activate this newly created environment:
    ```
    mamba activate queens
    ```
1. Install QUEENS inside the environment

    - For users :
        ```
        pip install -e .
        ```
    - For developers:
        ```
        pip install -e .[develop]
        ```
      The `-e` option (or `--editable`) ensures that QUEENS can be modified in the current location without the need to reinstall it.

1. If required, uninstall QUEENS within the activated environment:
    ```
    python uninstall queens
    ```

[↑ Contents](#contents)

## Start a *QUEENS* run
To start a *QUEENS* run with your *QUEENS* input file, run the following command in your [activated python environment](#usage):
```
queens --input <path-to-QUEENS-input> --output_dir <output-folder>
```

> **Note**: the output folder needs to be created by the user before starting the simulation.

More information:
```
queens --help
```
[↑ Contents](#contents)
## Further Topics

### Remote Computing
Remote computing is enabled via `ssh port forwarding`. Connecting via ssh from the local to the remote machine needs to work without password. Therefore, we need to copy the respective `id_rsa.pub`-keys to the respective machine:
```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@machine_you_would_like_to_access_passwordless
```
This command automatically checks whether you have already created an `id_rsa.pub`-key and copies it to the `authorized_keys`-file of
the remote machine you would like to get passwordless access to.

In case you do not have a `id_rsa.pub`-key on one of the machines, you can generate the key by running the subsequent command on the remote machine:
```batch
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
```

To enable passwordless access onto the localhost itself you need to copy the `ssh-key` of the localhost to the
   `authorized_keys` files by typing:
    ```
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    ```
To learn more about how ssh-port forwarding works click
[here](https://chamibuddhika.wordpress.com/2012/03/21/ssh-tunnelling-explained/).

If you have performed the aforementioned step, yet you are still asked for your password, this might be due to the following reasons:
 * permissions of the directory `~/.ssh` are incorrect. They have to be 700, so only the user itself is allowed to access the directory and the files in there.
   To set the permissions correctly, use the following command:
   ```batch
   chmod -R 700 ~/.ssh
   ```
 * write access to the home-directory on the remote machine has to be only allowed for the user.
   One possible way to set the permissions correctly is logging into the remote machine and using the following command:
   ```batch
   chmod 700 ~
   ```
   > **Note** that there are other valid choices. Refer to the manual of chmod for details.

[↑ Contents](#contents)

## Documentation
QUEENS uses [SPHINX](https://www.sphinx-doc.org/en/master/) to automatically build a html-documentation from docstring.
To build it, navigate into the doc folder with:
```bash
cd <queens-base_dir>/doc
```
and run:
```bash
make html
```
After adding new modules or classes to QUEENS, one needs to rebuild the autodoc index by running:
```bash
sphinx-apidoc -o doc/source pqueens -f -M
```
before the make command.

[↑ Contents](#contents)

## Run Test Suite
QUEENS has a couple of **unittests** and **integration test** which can be found under
 `<QUEENS_BaseDir>/pqueens/tests`.
In order to run local integration tests together with `BACI`, it is necessary that the user creates
**symbolic links** to the `BACI-executables`, which are then stored under
`<QUEENS_BaseDir>/config`. The links can be created by running the following commands:
```
ln -s <your/path/to/baci-release> <QUEENS_BaseDir>/config/baci-release
ln -s <your/path/to/post_ensight> <QUEENS_BaseDir>/config/post_ensight
ln -s <your/path/to/post_processor> <QUEENS_BaseDir>/config/post_processor
```
> **NOTE**: The workflows with BACI are tested with the BACI version [53abb9c91f90c909e559c8239f12757885dd81e3](https://gitlab.lrz.de/baci/baci/-/commit/53abb9c91f90c909e559c8239f12757885dd81e3)

The testing strategy is more closely described in [TESTING.md](pqueens/tests/README.md)
To run the test suite type:
```bash
pytest
```
To run the test suite with console output:
```bash
pytest -o log_cli=true --log-cli-level=INFO
```
For more info see the [pytest documentation](https://docs.pytest.org/en/latest/warnings.html).

[↑ Contents](#contents)
