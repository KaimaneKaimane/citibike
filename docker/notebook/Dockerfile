FROM python:3.7-slim-buster
ARG PROJECT

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    build-essential \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    vim \
    git \
    tk \
    tcl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the timezone.
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure Environment
ENV SHELL=/bin/bash \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    WORKDIR=/opt/project \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=$WORKDIR
ENV HOME=/home/$USER

RUN pip3 install jupyter
RUN pip3 install jupyter_contrib_nbextensions

# Install extensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextensions_configurator enable --user

# Install facets which do not have a pip or conda package at the moment
RUN cd /tmp && \
    git clone https://github.com/PAIR-code/facets.git && \
    cd facets && \
    jupyter nbextension install facets-dist/ --sys-prefix && \
    cd && \
    rm -rf /tmp/facets

# Enable the best extensions by default
RUN jupyter nbextension enable collapsible_headings/main && \
    jupyter nbextension enable navigation-hotkeys/main && \
    jupyter nbextension enable notify/notify && \
    jupyter nbextension enable execute_time/ExecuteTime && \
    jupyter nbextension enable hide_input_all/main && \
    jupyter nbextension enable toc2/main

# Setup notebook config
RUN mkdir -p $HOME/.jupyter/
COPY docker/notebook/jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py

RUN mkdir $WORKDIR
WORKDIR $WORKDIR

RUN pip3 install --upgrade pip

RUN pip3 install pipenv==2018.11.26

COPY docker/notebook/Pipfile $WORKDIR/Pipfile
COPY docker/notebook/Pipfile.lock $WORKDIR/Pipfile.lock

RUN pipenv install --system --deploy --dev

COPY . $WORKDIR/

# Run jupyter notebook
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser"]
