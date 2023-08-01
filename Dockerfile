ARG PYTHON_VERSION=3.8
ARG DISTRO=buster
ARG WORKSPACE=/monai_seg
ARG USERNAME=vscode

# ==================================================================================================

FROM python:$PYTHON_VERSION-$DISTRO as builder

ARG PYTHON_VERSION
ARG DISTRO
ARG WORKSPACE

# python
ENV PYTHONUNBUFFERED=1
# pip:
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=100
# poetry:
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

RUN pip install "poetry>=1.1.13,<2.0.0"

# Install python virtual environment in separate workspace
# Can later be copied to the active workspace, after the proper volumes have been mapped
WORKDIR $WORKSPACE

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-ansi    # -vvv for debug output if needed
# ==================================================================================================

FROM mcr.microsoft.com/vscode/devcontainers/python:$PYTHON_VERSION-$DISTRO

ARG PYTHON_VERSION
ARG DISTRO
ARG WORKSPACE
ARG USERNAME

ARG POLYAXON=0

# python:
ENV PYTHONUNBUFFERED=1
# pip:
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=100
# poetry:
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# if you need to install apt packages, do it here:
# RUN apt get update
# RUN apt install -y <your-package-list-here>

# add poetry version here to fix it
RUN pip install poetry

# [Optional] install the polyaxon cli outside of the virtual environment
RUN if [ $POLYAXON ] ; then pip3 install markupsafe==2.0.1 polyaxon-cli==0.6.1 ; fi

# ================================================================
# Get python virtual environment from builder
# Can later be copied to the active workspace, after the proper volumes have been mapped
WORKDIR /poetry-venv

COPY --from=builder /$WORKSPACE/.venv /poetry-venv/.venv
# ================================================================

# persist the bash history

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo "$SNIPPET" >> "/home/$USERNAME/.bashrc"


# Copy working-directory and copy files (overwritten if volumes are mapped)
# add additional build steps here, if needed
WORKDIR $WORKSPACE

COPY . .
