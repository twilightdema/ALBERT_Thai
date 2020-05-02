# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""BERT finetuning on classification tasks."""

'''
Run error analysis on fintuned model.
All error cases are logged.
Attention distribution in Transformer layers are also logged.
'''
import os
import time
import classifier_utils
import fine_tuning_utils
import modeling
import tokenization
import tensorflow.compat.v1 as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("cached_dir", None,
                    "Path to cached training and dev tfrecord file. "
                    "The file will be generated if not exist.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "albert_hub_module_handle", None,
    "If set, the ALBERT hub module to use.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("train_step", 1000,
                     "Total number of training steps to perform.")

flags.DEFINE_integer(
    "warmup_step", 0,
    "number of steps to perform linear learning rate warmup for.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "How many checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("optimizer", "adamw", "Optimizer to use")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": classifier_utils.ColaProcessor,
      "mnli": classifier_utils.MnliProcessor,
      "mismnli": classifier_utils.MisMnliProcessor,
      "mrpc": classifier_utils.MrpcProcessor,
      "rte": classifier_utils.RteProcessor,
      "sst-2": classifier_utils.Sst2Processor,
      "sts-b": classifier_utils.StsbProcessor,
      "qqp": classifier_utils.QqpProcessor,
      "qnli": classifier_utils.QnliProcessor,
      "wnli": classifier_utils.WnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.albert_config_file and not FLAGS.albert_hub_module_handle:
    raise ValueError("At least one of `--albert_config_file` and "
                     "`--albert_hub_module_handle` must be set")

  if FLAGS.albert_config_file:
    albert_config = modeling.AlbertConfig.from_json_file(
        FLAGS.albert_config_file)
    if FLAGS.max_seq_length > albert_config.max_position_embeddings:
      raise ValueError(
          "Cannot use sequence length %d because the ALBERT model "
          "was only trained up to sequence length %d" %
          (FLAGS.max_seq_length, albert_config.max_position_embeddings))
  else:
    albert_config = None  # Get the config from TF-Hub.

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name](
      use_spm=True if FLAGS.spm_model_file else False,
      do_lower_case=FLAGS.do_lower_case)

  label_list = processor.get_labels()

  tokenizer = fine_tuning_utils.create_vocab(
      vocab_file=FLAGS.vocab_file,
      do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file,
      hub_module=FLAGS.albert_hub_module_handle)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
  iterations_per_loop = FLAGS.iterations_per_loop
  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=int(FLAGS.save_checkpoints_steps),
      keep_checkpoint_max=0,
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = classifier_utils.model_fn_builder(
      albert_config=albert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.train_step,
      num_warmup_steps=FLAGS.warmup_step,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      task_name=task_name,
      hub_module=FLAGS.albert_hub_module_handle,
      optimizer=FLAGS.optimizer)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_predict:
  eval_examples = processor.get_dev_examples(FLAGS.data_dir)
  num_actual_eval_examples = len(eval_examples)
  if FLAGS.use_tpu:
    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on.
    while len(eval_examples) % FLAGS.predict_batch_size != 0:
      eval_examples.append(classifier_utils.PaddingInputExample())

  error_analysis_file = os.path.join(FLAGS.output_dir, "error_analysis.tf_record")
  classifier_utils.file_based_convert_examples_to_features(
      eval_examples, label_list,
      FLAGS.max_seq_length, tokenizer,
      error_analysis_file, task_name)

  tf.logging.info("***** Running error analysis on dev set*****")
  tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                  len(eval_examples), num_actual_eval_examples,
                  len(eval_examples) - num_actual_eval_examples)
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  error_analysis_drop_remainder = True if FLAGS.use_tpu else False
  error_analysis_input_fn = classifier_utils.file_based_input_fn_builder(
      input_file=error_analysis_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=error_analysis_drop_remainder,
      task_name=task_name,
      use_tpu=FLAGS.use_tpu,
      bsz=FLAGS.predict_batch_size)

  checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")
  result = estimator.predict(
      input_fn=error_analysis_input_fn,
      checkpoint_path=checkpoint_path)

  output_error_analysis_predict_file = os.path.join(FLAGS.output_dir, "error_analysis_test_results.tsv")
  output_error_analysis_submit_file = os.path.join(FLAGS.output_dir, "error_analysis_submit_results.tsv")
  with tf.gfile.GFile(output_error_analysis_predict_file, "w") as pred_writer,\
      tf.gfile.GFile(output_error_analysis_submit_file, "w") as sub_writer:
    sub_writer.write("index" + "\t" + "text_a" + "\t" + "text_b" + "\t" + "prediction" + "\t" + "label" + "\n")
    num_written_lines = 0
    tf.logging.info("***** Error analysis results *****")
    for (i, (example, prediction)) in\
        enumerate(zip(eval_examples, result)):
      probabilities = prediction["probabilities"]
      if i >= num_actual_eval_examples:
        break
      output_line = "\t".join(
          str(class_probability)
          for class_probability in probabilities) + "\n"
      pred_writer.write(output_line)

      if task_name != "sts-b":
        actual_label = label_list[int(prediction["predictions"])]
      else:
        actual_label = str(prediction["predictions"])
      sub_writer.write(example.guid + "\t" + str(example.text_a) + "\t" + str(example.text_b) + "\t" + str(actual_label) + "\t" + str(example.label) + "\n")
      num_written_lines += 1
  assert num_written_lines == num_actual_eval_examples

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("spm_model_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
