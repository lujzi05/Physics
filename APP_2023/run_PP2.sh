#!/bin/bash

path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "running PP2.py ..."
source $path/venv/bin/activate &&
cd $path &&

python3 $path/PP2.py