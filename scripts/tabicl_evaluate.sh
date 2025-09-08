#!/bin/bash

python ./src/evaluation/tabicl_evaluate.py \
    --datasets heloc \
    --data_dir ./datahub_inputs/data_raw \
    --output_dir ./datahub_outputs/tabicl \
    --n_jobs 4 \
    --sample_sizes 8 16 32 64 128 256 512 1024