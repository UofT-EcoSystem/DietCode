#!/bin/bash -e

cd $(dirname ${BASH_SOURCE[0]})

mkdir -p bert_base_uncased

wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin \
     -O bert_base_uncased/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt \
     -O bert_base_uncased/vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json \
     -O bert_base_uncased/config.json
