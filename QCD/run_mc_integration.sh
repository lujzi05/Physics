#!/bin/bash

path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $path &&
echo "$path" &&

if [ ! -d $path/venv ]; then
  echo "Creating virtual environment."
  python3 -m venv $path/venv &&
  source $path/venv/bin/activate &&

  pip3 install numpy
  pip3 install matplotlib
  pip3 install scipy
fi

echo "running mc_integration.py ..."
source $path/venv/bin/activate &&
cd $path &&

python3 $path/mc_integration.py
