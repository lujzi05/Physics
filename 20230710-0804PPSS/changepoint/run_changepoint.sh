#!/bin/bash

path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "running changepoint.py ..."
source $path/venv/bin/activate &&
cd $path &&

python3 $path/changepoint.py