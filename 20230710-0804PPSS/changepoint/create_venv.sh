#!/bin/bash

path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $path &&
echo "$path" &&

python3 -m venv $path/venv &&
source $path/venv/bin/activate &&

git clone https://github.com/jxx123/fireTS.git
cd fireTS
pip3 install -e .
cd ..

pip3 install -r requirements.txt