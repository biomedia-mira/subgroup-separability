FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as base

ARG PYTHON_VERSION=3.9

# Install python and linux packages. Using deadsnakes to easily select python version
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and add to path (this is equivalent to activating it)
ENV VIRTUAL_ENV=/opt/venv
RUN python${PYTHON_VERSION} -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install requirements in venv using poetry
# the bracket weirdness in poetry.lock is for the case where it doesnt exist
COPY pyproject.toml poetry.loc[k] /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install poetry \
    && cd /tmp/pip-tmp \
    && poetry config virtualenvs.create false \
    && poetry install --no-root \
    && rm -rf /tmp/pip-tmp \
    && chmod -R 777 $VIRTUAL_ENV

# Use the development stage with the vscode devcontainer
FROM base as dev

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


# prod is for reproducing results without the devcontainer (build image as executable)
FROM base as prod

WORKDIR /app
COPY . /app

CMD ["/bin/bash"]
