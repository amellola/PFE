# DeepAtlas MONAI App

This Repo implements the deepatlas-approach based on the [tutorial code](https://github.com/Project-MONAI/tutorials/blob/main/deep_atlas/deep_atlas_tutorial.ipynb) demonstrating simultaneous training of a registration and a segmentation network, implemented after [DeepAtlas](https://doi.org/10.1007/978-3-030-32245-8_47).

> Xu Z., Niethammer M. (2019) DeepAtlas: Joint Semi-supervised Learning of Image Registration and Segmentation. In: Shen D. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2019. MICCAI 2019. Lecture Notes in Computer Science, vol 11765. Springer, Cham. <https://doi.org/10.1007/978-3-030-32245-8_47>

## Table of Contents

* [Installation](#installation)
  * [On Host](#on-host)
    * [Poetry](#poetry)
    * [Polyaxon](#polyaxon)
  * [In Dev-Container](#in-dev-container)
    * [Requirements](#requirements)
      * [1. Docker](#1-docker)
      * [2. VSCode](#2-vscode)
    * [Configuration](#configuration)
      * [1. Docker-Compose](#1-docker-compose)
      * [2. OpenVPN](#2-openvpn)
      * [3. Polyaxon](#3-polyaxon)
      * [4. VSCode Dev-Containers/Remote-Containers](#4-vscode-dev-containersremote-containers)
* [Additional Configurations](#additional-configurations)
  * [VSCode Settings](#vscode-settings)
  * [Remote Development - Notes](#remote-development---notes)
* [Run in polyaxon](#run-in-polyaxon)
* [Usage](#usage)
* [Code structure](#code-structure)
* [Debugging Errors](#debugging-errors)
  * [Poetry stuck](#poetry-stuck)
* [Additional Explanations](#additional-explanations)
  * [Poetry Config and Commands](#poetry-config-and-commands)

## Installation

### On Host

This is for running and development on the host system, I'd recommend to use vscode [dev-containers](#in-dev-container).

#### Poetry

All dependencies are managed with poetry:

```bash
pipx install poetry
poetry install
```

Poetry install will create a `.venv` folder in the project and install all dependencies there.

With `poetry shell` you can activate the environment to work with the dependencies. Run `deactivate` to exit the environment.

Use `poetry run ...` to run any command in the virtual environment even when it is not activated.

To develop inside the environment, you need to set the python interpreter to `.venv/bin/python`. This should be default and recommended as it is specified in the `.vscode/settings.json` file.

For more infos on poetry check out the section on [poetry](#1-poetry)

#### Polyaxon

The polyaxon-cli need for the ifl-cluster is not compatible with newer python environments, as it uses some old dependencies, that many packages need newer versions from. To run it on your host machine you need to make sure to have `python3.8` installed

```bash
deactivate
pip install --user markupsafe==2.0.1 polyaxon-cli==0.6.1 # with multiple python versions try python3.8 -m pip ...
polyaxon config set --host=[ip_adress] --port [port]
polyaxon login --username [username]
```

For IFL:

* connect to IFL-VPN
* configure cli:

```bash
polyaxon config set --host=10.23.0.18 --port 31811
```

* login:

```bash
polyaxon login --username <username>
```

---
You can also use the login script for this process, so you don't have to repeat that all the time:

The first time you need to create the configuration. Polyaxon is configured by `polyaxon/config.cfg` for host and port and `polyaxon/pass.txt` for credentials. These files are read and used by the script `scripts/polyaxon-config.sh`.

The config is already set for the ifl cluster, if connected to the VPN

Add your credentials:

```bash
cp polyaxon/pass.txt.novalues polyaxon/pass.txt
code polyaxon/pass.txt
```

Run script:

```bash
sh scripts/polyaxon-config.sh
```

### In Dev-Container

If not using Dev-Container jump to [Polyaxon](#run-in-polyaxon)

#### Requirements

##### 1. Docker

You need the docker-engine and docker-compose:

* [Docker](https://docs.docker.com/engine/)
* [Docker-Compose](https://docs.docker.com/compose/)
* [Docker Desktop](https://docs.docker.com/desktop/)

I'd recommend installing docker-desktop if you are running this on an UI enabled machine (this installs both automatically).

If you want to run this on a remote computer, I'd recommend vscode with the remote-ssh extension and the regular docker engine with docker-compose. If working on a remote computer, you don't need docker on your local machine, only on the remote one.

##### 2. VSCode

Install [vscode](https://code.visualstudio.com/download)

[**Optional**]

If you want to run this on a remote computer, you need to install the `ms-vscode-remote.remote-ssh` extension and connect to the remote computer:

* (Recommended but optional) Generate ssh-key `ssh-keygen -t rsa -b 4096` (needs openssl installed)
* `Ctrl+shift+p` -> `Open SSH Configuration file` -> Edit file accordingly:

```bash
Host <identifier you give the machine>
    HostName <IP-Address or domain>
    User <username>
```

If you are using a ssh-key, you need to add `IdentityFile <path to key>` as well. On linux the path will probably be something like `~/.ssh/id_rsa` while on windows you should look under `C:/Users/<username>/.ssh/id_rsa` (you need the windows path even if you are running vscode in wsl).

* `Ctrl+shift+p` -> `Connect to Host` -> select the proper host -> enter password (if using ssh-key no need for that)

[**Required**]

Next install the extension: `ms-vscode-remote.remote-containers` (see [how](https://code.visualstudio.com/docs/editor/extension-marketplace))

#### Configuration

##### 1. Docker-Compose

In the `docker-compose.yaml` file, there are a few things, that can be customized to your project and system.

First if you don't want to run the vpn inside of the docker-container, you need to comment the entire service `vpn`. As the dev-container uses the `network_mode: "host"`, it has access to your vpn connection on the host-system, if you already have one running there. If you chose to run it inside the container, this will also allow you to make connections through the vpn on your host-system. (Access the polyaxon dashboard for example)

Next look at the service `polyaxon-dev`:

> :warning: **If you rename the service**:
>
> Make sure, you are also changing the config in `.devcontainer/devcontainer.json`, under `"service": "polyaxon-dev"`
>
> Same goes for changing the workspace-folder in `Dockerfile`, it is specified under `"workspaceFolder": "/polyaxon"` as well

Here you can specify a different python version and distro. The polyaxon-cli has been tested for `PYTHON_VERSION: 3.8` and `DISTRO: buster`, so be aware, that there might be problems with different versions and distributions.

> If on Mac, change to `DISTRO: bullseye`

You can also specify whether the `polyaxon-cli` should be pre-installed, by setting `POLYAXON: ...` to either 1 or 0.

Finally if you are not using a graphics card (or don't have one installed), you need to comment the marked part in the `docker-compose.yaml`, else the container won't start.

---
Check out the `makefiles/Makefile.docker` for some docker-compose commands, or run:

```bash
make help
```

##### 2. OpenVPN

The config files for the openvpn connection are located under `vpn-config`. The `ifl-cluster.ovpn` is already configured to work with the ifl-cluster. This will only work, if there is a second file `pass.txt`, which contains a valid username and password.

Edit the file:

```bash
cp vpn-config/pass.txt.novalues vpn-config/pass.txt
code vpn-config/pass.txt    # code for vscode, use other editor if it doesn't work
```

And enter your username and password, each in one line.

You can run the vpn container **by itself** with (to test or just have a vpn connection on your host-system, even if not using dev-containers):

```bash
make docker/run-vpn
```

and for live logs, to test if it's working:

```bash
make docker/logs
```

##### 3. Polyaxon

Polyaxon is configured by `polyaxon/config.cfg` for host and port and `polyaxon/pass.txt`. These files are read and used by the script `scripts/polyaxon-config.sh`, that is automatically executed with each container start.

Again before starting the container, you need to add your username and password:

```bash
cp polyaxon/pass.txt.novalues polyaxon/pass.txt
code polyaxon/pass.txt
```

After connected to the development container, you should be able to use the `polyaxon` command outside of the virtual environment. If the environment is activated and you want to exit it to run polyaxon, you need to run:

```bash
deactivate
```

If for some reason the config didn't run on container startup, or failed, you can run it manually with:

```bash
sh scripts/polyaxon-config.sh
```

##### 4. VSCode Dev-Containers/Remote-Containers

The `.devcontainer/devcontainer.json` specifies the settings for the development container, to run in vscode. Here you can adjust the settings to your liking as well.

Especially here you can add extensions that should be installed when the container is build, as well as some other interesting settings like port forwarding.

After everything is setup properly, you should only have to press `ctrl+shift+P` and then select `Rebuild and Reopen in Container`, either from `Dev-Containers:` or `Remote-Container:` depending on your version.

This will now build the docker image, install all dependencies and reopen the vscode instance inside the container.

## Additional Configurations

### VSCode Settings

In `.vscode/settings.json` you'll find some default settings for vscode to use, when this repo is opened. You can modify this file to your liking.

The same settings are also specified in the `.devcontainer/devcontainer.json` file, but these are overwritten by the other file.

### Remote Development - Notes

If you are developing on a remote server, through vscode-remote-ssh, you might run into problems with connecting to the vpn at the same time on your local machine (with the same client credentials), to check the polyaxon-dashboard.

This can be fixed, by tunneling your internet connection to the remote computer through ssh. This is fairly simple to do and enables you to use for example `localhost` to access servers running on you remote machine as well.

**1. Setup SSH-Tunnel:**

[**Linux**]

On linux you simple have to run the command:

```bash
ssh -D 9999 -N -f [<user>@]<host_name>
```

[**Windows**]

Windows requires you to use PuTTY:

* First install [PuTTY](https://www.putty.org/)
* Enter host details:

![enter host details](images/enter_host.png)

* If using ssh-key, enter key path:

![enter ssh-key path](images/ssh_key_auth.png)

* Tunnel port forwarding:

![enter tunnel port fowarding](images/tunnel_port_forward.png)

---

Find more info [here](https://linuxize.com/post/how-to-setup-ssh-tunneling/). You can forward specific ports, from you remote machine, or dynamic to allow full access to all interfaces.

---
Finally you need to connect to the just setup proxy on you local machine. This can be done globally, or what I prefer, by a browser extension, that can quickly switch back to your local system-proxy.

My choice is to work with chrome and use the extension [Proxy SwitchyOmega](https://chrome.google.com/webstore/detail/proxy-switchyomega/padekgcemlokbadohgkifijomclgjgif) but there are a lot of alternatives out there.

After installing you need to add your proxy profile. Here an example of how I use it:

![setup chrome proxy switcher](images/proxy_chrome.png)

> You can edit the bypass list if you want `localhost` or `127.0.0.1` to be routed through the proxy as well.

## Run in polyaxon

The polyaxonfiles are templates, that can be used, but should be modified. (*.group.yaml file has not been tested).

To generate the requirements.txt file from poetry you can use the command:

```bash
make update_req
```

This command needs an up-to-date poetry.lock file (holding all requirement specifications). So if that is not the case run:

```bash
poetry lock
```

Some of the polyaxon commands can be found in the `makefiles/Makefile.polyaxon` or by running:

```bash
make help
```

For creating a polyaxon project and using it, check out [this](https://dc.campar.in.tum.de/t/getting-started-with-polyaxon/215) page.

There are make commands for the process as well:

Init polyaxon project:

```bash
make polyaxon/create name=<your_project_name>   # if not created in the dashboard yet
make polyaxon/init name=<your_project_name> # will create local config files in .polyaxon
```

To start a single experiment:

```bash
make polyaxon/run
```

To start and stop a tensorboard for an experiment:

```bash
make polyaxon/board xp=<experiment id>
```

```bash
make polyaxon/stop_board xp=<experiment id>
```

## Usage

The in the main file specified cli is implemented to run with a lot of different options, to train a segmentation and a registration network via the deepatlas approach.

```bash
> python -m deepatlas.main --help
Usage: python -m deepatlas.main [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                                [ARGS]...]...

  Implementation of the DeepAtlas approach to segmentation and registration of
  medical images.

Options:
  --polyaxon   enable polyaxon functions
  -d, --debug  enable debug mode
  --help       Show this message and exit.

Commands:
  infer      Only run inference on images with pre-trained models.
  metrics    Calc metrics from ground truth and predictions.
  train      Train the model.
  transform  Only transform images and labels and save.
```

The makefiles, specifically `makefiles/Makefile.run` offers some example commands to run training, inference and metrics generation on the oasis dataset (if specified will be downloaded automatically) or an custom dataset (tested with ultrasound images), that needs to be prepared.

## Code structure

You'll find the important code in the `deepatlas/lib` folder. It's structured into:

* config

> Takes care of the basic configurations for each model. Main hyper parameters, network, additional configuration
> Edit here to change the network(s)

* dataloaders

> Implements the data-loading logic, to change the way data is loaded edit here

* infers

> Holds the inference logic and configuration for pre-, inverse- and post-transforms, as well as the inferer used

* trainers

> Holds the train logic and configuration for training-transforms, optimizer and used loss function(s), the training loop is implemented here

* losses

> here, if needed, specific loss classes can be implemented to import in trainers. Specifically for deepatlas, where we need multiple loss functions, these are implemented here

* metrics

> This offers a place to put metrics calculation for importing

* transforms

> Offers a space to put different transform configurations for importing if trying multiple for example

## Debugging Errors

If something doesn't work, try to test it one by one:

* poetry on your host system
* docker by building manually and trying to launch the containers individually
* vpn by checking the docker-logs or connecting on your host-system

Check the Makefiles for a collection of useful commands!

### Poetry stuck

* `torch` and `werkzeug` tend to take really long without giving any output
* check if you commented the graphics card settings in the `docker-compose.yaml` file, if you don't have a nvidia graphics card
* sometimes there might be networking issues when building while the vpn-container is running. Try stopping the container on the host system, by running `docker stop vpn`.

## Additional Explanations

### Poetry Config and Commands

You can edit the `pyproject.toml` file under `[tool.poetry]` to your liking (name, version, ...).

Next install your dependencies with poetry (this follows the same version-constraint and extension rules as pip):

```bash
poetry add <package-name> "<package-name>==<version>"
```

For adding development dependencies:

```bash
poetry add -D <package-name> "<package-name>==<version>"
```

Other poetry settings can be set in `poetry.toml`. But if you don't know the tool, I'd suggest to leave the defaults.

> If your system gives you errors on installing the dependencies, add `--lock` to the command, so it only adds the requirement without installing

After connected to the development container, you can activate the virtual environment with:

```bash
make venv
```

or (`poetry shell`)
