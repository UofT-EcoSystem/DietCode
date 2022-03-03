#!/bin/bash -e

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/..

AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS:-1500}
source ${PROJECT_ROOT}/environ/activate_dev.sh

cd ${PROJECT_ROOT}/ops/dense
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_dietcode.py::test_train_BERT_H768
cp dietcode_autosched_dense_16xTxIx768.json saved_schedules_G4
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_dietcode.py::test_train_BERT_H3072
cp dietcode_autosched_dense_16xTx768x3072.json saved_schedules_G4

mkdir -p 2_4-saved_artifacts
cd 2_4-saved_artifacts
cp ../*.csv .

cd ${PROJECT_ROOT}/ops/batch_matmul
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_dietcode.py::test_train_dynT_1
cp dietcode_autosched_batch_matmul_nt_192xTx64xT.json saved_schedules_G4
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_dietcode.py::test_train_dynT_2
cp dietcode_autosched_batch_matmul_nt_192xTxTx64.json saved_schedules_G4

mkdir -p 2_4-saved_artifacts
cd 2_4-saved_artifacts
cp ../*.csv .

cd ${PROJECT_ROOT}/networks/bert
./download_bert_base_uncased.sh
pytest -s test_dietcode.py::test_infer_dynT

mkdir -p 2_4-saved_artifacts
cd 2_4-saved_artifacts
cp ../*.csv .
