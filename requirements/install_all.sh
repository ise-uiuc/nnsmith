#!/bin/bash

set -ex

cd ./requirements

# detect cuda
# check if nvcc can work and return true or false
check_nvcc() {
    if command -v nvcc >/dev/null; then
        return 0
    else
        return 1
    fi
}

pip install pip --upgrade # upgrade pip

find . -name '*.txt' | while read -r dep_file; do
    if [ -f "$dep_file" ]; then
        # skip tensorrt if cuda is not available
        if [[ "$dep_file" == *"tensorrt"* ]] && ! check_nvcc; then
            echo "Skipping tensorrt"
            continue
        fi
        if [[ "$dep_file" == *"sys"* ]]; then
            # Files under sys should be nightly releases
            pip install -r "$dep_file" --upgrade --pre
        else
            pip install -r "$dep_file"
        fi
    fi
done
