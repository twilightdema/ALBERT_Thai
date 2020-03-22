
SQUAD_DATA_DIR=squad_data
OUTPUT_DIR_BASE="output_squad_v2"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/output"

ALBERT_CONFIG_FILE="pretrained_models/albert_base/albert_config.json"
SPM_VOCAB_PATH="pretrained_models/albert_base/30k-clean.vocab"
SPM_MODEL_PATH="pretrained_models/albert_base/30k-clean.model"
INIT_CHECKPOINT="pretrained_models/albert_base/model.ckpt-best"

python3 run_squad_v2.py \
  --albert_config_file=${ALBERT_CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR} \
  --train_file=${SQUAD_DATA_DIR}/train-v2.0.json \
  --predict_file=${SQUAD_DATA_DIR}/dev-v2.0.json \
  --train_feature_file=${SQUAD_DATA_DIR}/train-v2.0.tfrecord \
  --predict_feature_file=${SQUAD_DATA_DIR}/dev-v2.0.tfrecord \
  --predict_feature_left_file=${SQUAD_DATA_DIR}/dev-v2.0.left.tfrecord \
  --init_checkpoint=${INIT_CHECKPOINT} \
  --spm_model_file=${SPM_MODEL_PATH} \
  --do_lower_case \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64 \
  --do_train=true \
  --do_predict=true \
  --train_batch_size=16 \
  --predict_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=1628 \
  --iterations_per_loop=1628 \
  --n_best_size=20 \
  --max_answer_length=30