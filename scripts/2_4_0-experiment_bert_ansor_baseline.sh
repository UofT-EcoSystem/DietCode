#!/bin/bash -e

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/..

AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS:-1000}
source ${PROJECT_ROOT}/environ/activate_base.sh

AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_dietcode.py::test_train_dynT
cp *.json saved_schedules_G4
