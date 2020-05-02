#!/bin/bash
# This is a convenience script for evaluating ALBERT on the GLUE benchmark.
#
# By default, this script uses a pretrained ALBERT v1 BASE model, but you may
# use a custom checkpoint or any compatible TF-Hub checkpoint with minimal
# edits to environment variables (see ALBERT_HUB_MODULE_HANDLE below).
#
# This script does fine-tuning and evaluation on 8 tasks, so it may take a
# while to complete if you do not have a hardware accelerator.

set -ex

#python3 -m venv $HOME/albertenv
#. $HOME/albertenv/bin/activate

GLUE_DATA_DIR=glue_data
OUTPUT_DIR_BASE="output_glue"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/output"

# To start from a custom pretrained checkpoint, set ALBERT_HUB_MODULE_HANDLE
# below to an empty string and set INIT_CHECKPOINT to your checkpoint path.
ALBERT_HUB_MODULE_HANDLE=""
ALBERT_CONFIG_FILE="pretrained_models/albert_base/albert_config.json"
SPM_VOCAB_PATH="pretrained_models/albert_base/30k-clean.vocab"
SPM_MODEL_PATH="pretrained_models/albert_base/30k-clean.model"
INIT_CHECKPOINT="pretrained_models/albert_base/model.ckpt-best"
#pip3 install --upgrade pip
#pip3 install numpy
#pip3 install -r requirements.txt

function run_task() {
  COMMON_ARGS="--output_dir="${OUTPUT_DIR}/$1" --data_dir="${GLUE_DATA_DIR}" --vocab_file="${SPM_VOCAB_PATH}" --spm_model_file="${SPM_MODEL_PATH}" --do_lower_case --max_seq_length=128 --optimizer=adamw --task_name=$1 --warmup_step=$2 --learning_rate=$3 --train_step=$4 --save_checkpoints_steps=$5 --iterations_per_loop=$5 --train_batch_size=$6"
  python3 -m run_classifier_error_analysis \
      ${COMMON_ARGS} \
      --albert_hub_module_handle="${ALBERT_HUB_MODULE_HANDLE}" \
      --albert_config_file="${ALBERT_CONFIG_FILE}"
}

run_task SST-2 1256 1e-5 20935 2093 32
run_task MNLI 1000 3e-5 40000 4000 32
run_task CoLA 320 1e-5 5336 533 16
run_task QNLI 1986 1e-5 33112 3311 32
run_task QQP 1000 5e-5 56000 5600 32
run_task RTE 200 3e-5 800 80 32
run_task STS-B 214 2e-5 3598 359 16
run_task MRPC 200 2e-5 800 80 32
