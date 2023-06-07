#!/bin/bash

path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $path &&
echo "$path" &&

python3 -m venv $path/venv &&
source $path/venv/bin/activate &&

pip3 install -r requirements.txt