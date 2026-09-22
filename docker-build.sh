#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    --build-arg VERSION="${VERSION}" \
    -t motionized-audio:${VERSION} \
    -t motionized-audio:latest .

echo "Built motionized-audio:${VERSION} (also tagged :latest)"
