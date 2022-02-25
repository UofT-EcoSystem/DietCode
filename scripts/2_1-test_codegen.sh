#!/bin/bash -e

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/..

cd ${PROJECT_ROOT}/tests
source ${PROJECT_ROOT}/environ/activate_dietcode.sh
python3 -m pytest codegen
