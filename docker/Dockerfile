FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Basic setting library
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install --yes --no-install-recommends \
    sudo \
    git \
    tig \
    cmake \
    pkg-config \
    unzip \
    wget \
    build-essential \
    libssl-dev\
    libbz2-dev\
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    curl \
    libsm6 \
    libgtk2.0-dev \
    vim \
    less \
    pv \
    zlib1g-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    liblzma-dev \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Make sudo user
ARG USERNAME
RUN useradd ${USERNAME} -G sudo -s /bin/bash -m  && \
    echo ${USERNAME}' ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV HOME /home/${USERNAME}

# Install pyenv & install python
ENV PYTHON_VERSION 3.7.2
ENV PYENV_ROOT /home/${USERNAME}/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

# Install python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip setuptools && \
    pip install -r /tmp/requirements.txt

# Install nvidia/apex
# https://github.com/NVIDIA/apex
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
WORKDIR /home/${USERNAME}

# Install kaggle command
COPY kaggle_apikey.json /tmp/kaggle.json
RUN mkdir $HOME/.kaggle && \
    cp /tmp/kaggle.json $HOME/.kaggle/kaggle.json && \
    chmod 600 $HOME/.kaggle/kaggle.json

# Install fish shell
RUN sudo apt-get update && \
    sudo apt-get -y install software-properties-common && \
    sudo apt-add-repository ppa:fish-shell/release-3 && \
    sudo apt-get -y install fish && \
    curl https://git.io/fisher --create-dirs -sLo ~/.config/fish/functions/fisher.fish

# change the default shell
ENV SHELL /usr/bin/fish

# install your fish plugins
RUN /usr/bin/fish -c "fisher add oh-my-fish/theme-eclm jethrokuan/z 0rax/fish-bd oh-my-fish/plugin-peco"

# config and clean up
RUN sudo apt-get clean && sudo apt-get autoremove
