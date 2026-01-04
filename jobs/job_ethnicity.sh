#!/bin/bash
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j
#PJM -o ethnicity_%j.out
#PJM --name ig2-ethnicity
#PJM --mail-list ripitoaa@gmail.com
#PJM -m b,e

set -eu

WORK_DIR="/work/gb20/b20117/IG2-Bias-Neurons"
CONTAINER="/work/gb20/b20117/pytorch.sif"

echo "=============================================="
echo "Original IG2 - Ethnicity (Black/White) Analysis"
echo "Job ID: ${PJM_JOBID:-local}"
echo "Start: $(date)"
echo "=============================================="

cd "${WORK_DIR}"
source /etc/profile.d/modules.sh
module load singularity/3.7.3

# Stage 1: Compute IG² scores
echo "Stage 1: Computing Integrated Gradients..."
singularity exec --nv \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    --env HF_HOME="${WORK_DIR}/.cache/huggingface" \
    "${CONTAINER}" \
    python bias_neuron_src/1_analyze_mlm_bias.py \
        --data_path bias_neuron_data \
        --demographic_dimension Ethnicity \
        --demographic1 black \
        --demographic2 white \
        --modifier negative \
        --bert_model_path bert-base-uncased \
        --output_dir results/ethnicity \
        --do_lower_case \
        --gpus 0 \
        --get_ig2_gold \
        --get_base \
        --get_ig2_gold_gap_filtered \
        --debug 24

# Stage 2: Extract bias neurons
echo "Stage 2: Extracting bias neurons..."
singularity exec --nv \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${CONTAINER}" \
    python bias_neuron_src/2_get_bn_bias_v2_ethnicity_white_black.py

echo "=============================================="
echo "Completed at: $(date)"
echo "=============================================="
