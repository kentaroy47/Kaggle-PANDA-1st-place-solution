#!/bin/bash

docker build --build-arg USERNAME=`whoami` -t kaggle -f docker/Dockerfile ./docker
