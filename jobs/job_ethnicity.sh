#!/bin/bash
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=08:00:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j
#PJM -o ig2_ethnicity_%j.out
#PJM --name ig2-ethnicity
#PJM --mail-list ripitoaa@gmail.com
#PJM -m b,e

# Original IG² Analysis: Ethnicity (black vs white)
# Compare with SignedIG² results from proanti-SignedIG2

set -eu

source /etc/profile.d/modules.sh
module load singularity/3.7.3

CONTAINER="/work/gb20/b20117/pytorch.sif"
PROJECTDIR="/work/gb20/b20117/IG2-Bias-Neurons"
MODEL="bert-base-cased"

cd "${PROJECTDIR}"

echo "=============================================="
echo "Original IG² Analysis: Ethnicity"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Start time: $(date)"
echo ""

# Use converted data from proanti-SignedIG2
DATA_FILE="/work/gb20/b20117/proanti-SignedIG2/number/data/converted/ethnicity_negative.json"

singularity exec --nv \
    --bind "${PROJECTDIR}:${PROJECTDIR}" \
    --bind "/work/gb20/b20117/proanti-SignedIG2:/work/gb20/b20117/proanti-SignedIG2" \
    --env HF_HOME="${PROJECTDIR}/.cache/huggingface" \
    "${CONTAINER}" python scripts/run_ig2.py \
        --model "${MODEL}" \
        --data_file "${DATA_FILE}" \
        --d1 black \
        --d2 white \
        --output_dir "results/${MODEL}/ethnicity_black_white" \
        --num_steps 50

echo ""
echo "=============================================="
echo "Ethnicity Analysis Complete!"
echo "=============================================="
echo "End time: $(date)"
