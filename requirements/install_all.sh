#!/bin/bash

set -ex

cd ./requirements

for dep_file in $(find . -name '*.txt')
do
  if [ -f "$dep_file" ]; then
      pip install -r "$dep_file" --upgrade
  fi
done
