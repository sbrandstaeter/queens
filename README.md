# QUEENS
This repository contains the code of *QUEENS*, a general purpose framework for Uncertainty Quantification,
Physics-Informed Machine Learning, Bayesian Optimization, Inverse Problems and Simulation Analytics on distributed computer
systems.

## Contents
- [QUEENS](#queens)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
    - [Anaconda](#anaconda)
    - [Git](#git)
    - [MongoDB](#mongodb)
    - [Optional: Singularity](#optional-singularity)
  - [Installation](#installation)
  - [Further Topics](#further-topics)
    - [Remote Computing](#remote-computing)
    - [LNM-specific issues](#lnm-specific-issues)
  - [Documentation](#documentation)
  - [Run Test Suite](#run-test-suite)
    - [GitLab Test Machine](#gitlab-test-machine)


## Prerequisites
There are various prerequisites for QUEENS such as (an appropriately configured) Git, Anaconda, and MongoDB.

[↑ Contents](#contents)

### Anaconda
[Install](http://docs.anaconda.com/anaconda/install/linux/) the latest version of [Anaconda](https://www.anaconda.com/) with Python 3.x.
 *Anaconda* is an open-source Python distribution and provides a [virtual environment manager named *Conda*](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) with many popular data science Python packages. In the following, we will provide some of the most important commands when using Anaconda.

- Create a new Anaconda environment, e.g., using Python 3.8:
```
conda create -n <name_of_your_environment> python=3.8
```

- List all packages linked into an Anaconda environment:
```
conda list -n <name_of_your_environment>
```

- Activate an environment:
```
source activate <name_of_your_environment>
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
    pre-commit install --hook-type prepare-commit-msg
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
     [GitLab documentation](https://gitlab.lrz.de/help/ssh/README).

[↑ Contents](#contents)

### MongoDB
QUEENS uses a [MongoDB database](https://www.mongodb.com/) for data handling. Therefore, QUEENS requires certain write and access rights. MongoDB does not necessarily have to run on the same machine as QUEENS, although this is the default case. In certain situations, though, it might make sense to have the database running on a different computer and connect to the database via port-forwarding.

For installation on various OS, the following hints might be useful:

- Installation instructions for **macOS** can be found [here](https://docs.mongodb.com/master/tutorial/install-mongodb-on-os-x/?_ga=2.181134695.1149150790.1494232459-1730069423.1494232449)

- When using **CentOS 7**, MongoDB can be installed as follows:
```
sudo yum install mongodb-org
```

- Installation instructions for **Ubuntu** can be found [here](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/).

- Installation instructions for **Fedora 22** can be found [here](https://blog.bensoer.com/install-mongodb-3-0-on-fedora-22/).

The commands for starting and potentially stopping MongoDB read as follows:

- Start MongoDB:
    ```
    sudo systemctl start mongod
    ```

- If required, stop MongoDB:
    ```
    sudo systemctl stop mongod
    ```

- And then potentially restart again:
    ```
    sudo systemctl restart mongod
    ```

- (Optional) For MacOS, you have to run:
    ```
    mongod --dbpath <path_to_your_database>`
    ```
    > Note: It might be required to edit the sudoers file `/etc/sudoers.d/` (together with your administrator) to get execution rights.



Note that the MongoDB config file might be edited such that it allows for connections from anywhere (probably not the safest option, but ok for now). If you followed the standard installation instructions, you should edit the config file using
```
sudo vim /etc/mongod.conf
```
And comment out the `bindIp` like shown
```
# network interfaces
net:
port: 27017
# bindIp: 127.0.0.1  <-- comment out this line
```
>Note: The machine running the MongoDB database does not need to be the same as either the machine running QUEENS. However, the other machines need to be able to connect to the MongoDB via tcp to write data. Within an internal network, all machines need to be connected to that internal network.

>Note: If you are experiencing SELinux warnings follow the solution [here](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-red-hat/#optional-suppress-ftdc-warnings).
[↑ Contents](#contents)

### Optional: Singularity

Singularity containers are well suited for being used with QUEENS. If you are interested in using Singularity containers for your computations,  please make sure you either have an existing Singularity image `singularity_image.sif` available or `Singularity` installed on your local machine. For instance, in CentOs7, singularity can directly be installed from the repository via:
```
sudo yum install singularity
```
After the installation, execute the following command once on your workstation:
```
sudo singularity config fakeroot --add <your_username>
```
To check that the command was successful, you can
1. show the generated `/etc/subuid`
    ```bash
    cat /etc/subuid
    ```
    which should return something like `1000:4294836224:65536`.
For more information please refer to the singularity documentation for [user](https://sylabs.io/guides/3.5/user-guide/fakeroot.html) and [admin](https://sylabs.io/guides/3.5/admin-guide/user_namespace.html#config-fakeroot) on the fakeroot option of singularity.

Make sure that after the installation process your `.bashrc` file contains
```
export SINGULARITY_BIND=/opt:/opt,/bin:/bin,/etc:/etc,/lib:/lib,/lib64:/lib64,/lnm:/lnm
export SINGULARITYENV_APPEND_PATH=$PATH
export SINGULARITYENV_APPEND_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
```
(Generally this should be generated automatically. Without `lnm` on external PCs.)

Assuming singularity is already installed, an image for a QUEENS run can be build (in the environment where QUEENS is installed) using the QUEENS CLI with the command:
```
queens-build-singularity
```
[↑ Contents](#contents)


## Installation
*QUEENS* needs only to be installed once on the local machine, allowing already for both local and remote computing.

1. After having [Git](#configuration-of-git) availabe and appropriately configured, this repository might be cloned to your local machine:
    ```
    git clone git@gitlab.lrz.de:queens_community/queens.git <target-directory>
    ```
1. Assuming that Anaconda is installed on your local machine, create a QUEENS environment:
    ```
    cd <your-path-to-QUEENS>
    conda env create
    ```
    With this, all required third party libraries will be installed.

    advanced: for a custom environment name
    `conda env create -f  <your-path-to-QUEENS>/environment.yml --name <your-custom-queens-env-name>`
1. Activate this newly created environment:
    ```
    conda activate queens
    ```
1. Install QUEENS inside the environment:
    - For simple use (in the QUEENS directory):
        ```
        pip install -e .
        ```
    - For developers (in the QUEENS directory):
        ```
        pip install -e .[develop]
        ```
    The `-e` option (or `--editable`) ensures that QUEENS can be modified in the current location without the need to reinstall it.
1. If required, uninstall QUEENS within the activated environment:
    ```
    python uninstall queens
    ```

Updates from time to time are recommended:

- Update your Python packages the easy (default) way:
   ```
   cd <your-path-to-QUEENS>
  conda env update
   ```
- Update your Python packages in a more advanced way:
   ```
   conda env update --verbose --name <your-custom-queens-env-name> -f <your-path-to-QUEENS>/environment.yml
   ```

- Update Python version of your Conda environment: to be in sync with the latest Python version recommended for QUEENS, act in sync with the latest changes on the  `master` branch and follow the normal update procedure described above. This will keep both your environment-related file `environment.yml` and your environment up to date.

Furthermore, you might be interested in using Jupyter notebooks:

- Use conda environments in Jupyter notebooks: to get your conda environments into your Jupyter kernel, run the following command:

   ```
   conda install nb_conda_kernels
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
   Note that there are other valid choices. Refer to the manual of chmod for details.

[↑ Contents](#contents)

### LNM-specific issues
In case you want to run simulations on remote computing machines (e.g., cluster), you need enable access from the remote to the localhost at `port27017`.By default the firewall software `firewalld` blocks every incoming request. Hence, to enable a connections, we have
add so called rules to `firewalld` in order to connect to the database.
1.  First check the current firewall rules by typing:
    ```bash
    sudo firewall-cmd --list-all
    ```
1. If there is no rule in place which allows you to connect to port 27017, you have to add an exception for the *master-node* of the clusters you want work with:
    ```bash
    sudo firewall-cmd --zone=work --add-rich-rule 'rule family=ipv4 source address=<IP-address-of-cluster-master-node> port port=27017 protocol=tcp accept' --permanent
    ```
    Some LNM specific IP addresses are:
    - Schmarrn: 129.187.58.24
    - Bruteforce (master node global IP): 129.187.58.13
    - Deep (master node global IP): 129.187.58.20

1. To apply the changes run:
    ```
    sudo firewall-cmd --reload
    ```
1. To enable passwordless access onto the localhost itself you need to copy the `ssh-key` of the localhost to the
   `authorized_keys` files by typing:
    ```
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    ```

In case you want to use queens with `bruteforce` at the beginning of the first execution of queens on `bruteforce` your `.bashrc`-file is manipulated. Three lines with:
```
export SINGULARITY*
```
are added. However, you need to manually modify the line:
```
export SINGULARITYENV_APPEND_LD_LIBRARY_PATH=/something/has/been/added/automatically
```
and replace the `/something/has/been/added/automatically` by the output of:
```
echo $LD_LIBRARY_PATH
```
while you are regularly logged in on bruteforce with your account.

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
ln -s <your/path/to/post_drt_monitor> <QUEENS_BaseDir>/config/post_drt_monitor
ln -s <your/path/to/post_drt_ensight> <QUEENS_BaseDir>/config/post_drt_ensight
ln -s <your/path/to/post_processor> <QUEENS_BaseDir>/config/post_processor
```
The testing strategy is more closely described in [TESTING.md](TESTING.md)
To run the test suite type:
```bash
pytest pqueens/tests -W ignore::DeprecationWarning
```
For more info see the [pytest documentation](https://docs.pytest.org/en/latest/warnings.html).

[↑ Contents](#contents)

### GitLab Test Machine
To get started with Gitlab checkout the help section [here](https://gitlab.lrz.de/help/ci/README.md).
CI with with gitlab requires the setup of so called runners that build and test the code.
To turn a machine into a runner, you have to install some software as described
[here](https://docs.gitlab.com/runner/install/linux-repository.html)
The steps are as follows:
1. For RHEL/CentOS/Fedora run
    ```bash
    curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh | sudo bash
    ```
1. To install run:
    ```bash
    sudo yum install gitlab-runner
    ```
1. Next, you have to register the runner with your gitlab repo as described
[here](https://docs.gitlab.com/runner/register/index.html).

[↑ Contents](#contents)
