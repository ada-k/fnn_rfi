#!/usr/bin/env bash
# Build and run the AOFlagger container.
# Usage:
#  ./build_and_run.sh build    # builds the docker image
#  ./build_and_run.sh run /local/data/dir  # runs container with /local/data/dir mounted at /data

set -euo pipefail

IMAGE_NAME="aoflagger:local"
CMD=${1:-}

if [[ "$CMD" == "build" ]]; then
  docker build -t ${IMAGE_NAME} .
  echo "Built ${IMAGE_NAME}"
  exit 0
fi

if [[ "$CMD" == "run" ]]; then
  DATA_DIR=${2:-"$PWD"}
  docker run --rm -it -v "$DATA_DIR":/data ${IMAGE_NAME}
  exit 0
fi

echo "Usage: $0 build|run [local_data_dir]"
exit 2
