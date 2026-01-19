#!/bin/bash
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j
#PJM -o ig2_test_%j.out
#PJM --name ig2-test
#PJM --mail-list ripitoaa@gmail.com
#PJM -m b,e

# Quick test: 2 samples only to verify code works

set -eu

source /etc/profile.d/modules.sh
module load singularity/3.7.3

CONTAINER="/work/gb20/b20117/pytorch.sif"
PROJECTDIR="/work/gb20/b20117/IG2-Bias-Neurons"
MODEL="bert-base-cased"

cd "${PROJECTDIR}"

echo "=============================================="
echo "Original IG² Test Run"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Start time: $(date)"
echo ""

DATA_FILE="/work/gb20/b20117/proanti-SignedIG2/number/data/converted/gender_negative.json"

singularity exec --nv \
    --bind "${PROJECTDIR}:${PROJECTDIR}" \
    --bind "/work/gb20/b20117/proanti-SignedIG2:/work/gb20/b20117/proanti-SignedIG2" \
    --env HF_HOME="${PROJECTDIR}/.cache/huggingface" \
    "${CONTAINER}" python scripts/run_ig2.py \
        --model "${MODEL}" \
        --data_file "${DATA_FILE}" \
        --d1 female \
        --d2 male \
        --output_dir "results/${MODEL}/test" \
        --num_steps 10 \
        --max_samples 2

echo ""
echo "Test Complete!"
echo "End time: $(date)"
