#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    --build-arg VERSION="${VERSION}" \
    -f Dockerfile.gpu \
    -t motionized-audio-gpu:${VERSION} \
    -t motionized-audio-gpu:latest .

echo "Built motionized-audio-gpu:${VERSION} (also tagged :latest)"
