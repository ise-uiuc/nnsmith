#!/bin/bash

set -ex

cd ./requirements

# detect cuda
# check if nvcc can work and return true or false
check_nvcc() {
    nvcc -V > /dev/null 2>&1
    return $has_cuda
}

for dep_file in $(find . -name '*.txt')
do
  if [ -f "$dep_file" ]; then
      # skip tensorrt if cuda is not available
      if [[ "$dep_file" == *"tensorrt"* ]] && ! check_cuda; then
          echo "Skipping tensorrt"
          continue
      fi

      UPGRADE_PACKAGES=""
      if [[ "$dep_file" == *"sys"* ]]; then
          # Files under sys should be nightly releases
          UPGRADE_PACKAGES="--upgrade"
      fi
      pip install -r "$dep_file" $UPGRADE_PACKAGES
  fi
done
