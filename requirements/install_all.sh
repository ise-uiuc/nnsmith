#!/bin/bash

set -ex

cd ./requirements

for dep_file in $(find . -name '*.txt')
do
  if [ -f "$dep_file" ]; then
      pip install --upgrade -r "$dep_file"
  fi
done
