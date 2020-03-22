
RACE_DATA_DIR=race_data
OUTPUT_DIR_BASE="output_race"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/output"

ALBERT_CONFIG_FILE="pretrained_models/albert_base/albert_config.json"
SPM_VOCAB_PATH="pretrained_models/albert_base/30k-clean.vocab"
SPM_MODEL_PATH="pretrained_models/albert_base/30k-clean.model"
INIT_CHECKPOINT="pretrained_models/albert_base/model.ckpt-best"

python3 run_race.py \
  --albert_config_file=${ALBERT_CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR} \
  --train_file=${RACE_DATA_DIR}/train.tfrecord \
  --eval_file=${RACE_DATA_DIR}/dev.tfrecord \
  --data_dir=${RACE_DATA_DIR} \
  --init_checkpoint=${INIT_CHECKPOINT} \
  --spm_model_file=${SPM_MODEL_PATH} \
  --max_seq_length=512 \
  --max_qa_length=128 \
  --do_train=true \
  --do_eval=true \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --learning_rate=1e-5 \
  --train_step=12000 \
  --warmup_step=1000 \
  --save_checkpoints_steps=1200 \
  --iterations_per_loop=1200
  