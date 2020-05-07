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

RUN mkdir $WORKDIR
WORKDIR $WORKDIR

RUN pip3 install --upgrade pip

RUN pip3 install pipenv==2018.11.26

COPY docker/project/Pipfile $WORKDIR/Pipfile
COPY docker/project/Pipfile.lock $WORKDIR/Pipfile.lock

RUN pipenv install --system --deploy --dev

COPY code $WORKDIR/

CMD ["python"]