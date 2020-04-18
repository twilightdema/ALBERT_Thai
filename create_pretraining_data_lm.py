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
# Lint as: python2, python3
# coding=utf-8
"""Create Language Model TF examples for ALBERT (Decoder-Only)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
import tokenization
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string("input_file_mode", "r",
                    "The data format of the input file.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", True,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 256, "Maximum sequence length.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

class LMTrainingInstance(object):
  """A single training instance."""

  def __init__(self, tokens, token_boundary):
    self.tokens = tokens
    self.token_boundary = token_boundary

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "token_boundary: %s\n" % (" ".join(
        [str(x) for x in self.token_boundary]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    output_files):
  """Create TF example files from `LMTrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    print('Saving instance ' + str(inst_index))
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    
    # For LM, input mask is 2D Array with Transformer Decoder masking style.
    # In order to save space, we will expand the data to 2D when feeding to model.
    # Here we just need to store ID of sequence so we can reconstruct the 2D map corresponding to the sequence later,
    input_mask = [1] * len(input_ids)

    token_boundary = list(instance.token_boundary)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      token_boundary.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["token_boundary"] = create_int_feature(token_boundary)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              short_seq_prob, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    line_num = 0
    with tf.gfile.GFile(input_file, FLAGS.input_file_mode) as reader:
      while True:
        print('Reading line ' + str(line_num))
        line = reader.readline()
        if not FLAGS.spm_model_file:
          line = tokenization.convert_to_unicode(line)
        if not line:
          break
        if FLAGS.spm_model_file:
          line = tokenization.preprocess_text(line, lower=FLAGS.do_lower_case)
        else:
          line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)
        line_num = line_num + 1

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  print('all_documents length = ' + str(len(all_documents)))

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for document_index in range(len(all_documents)):
    print('Creating instance for doc ' + str(document_index))
    instances.extend(
        create_instances_from_document(
            all_documents, document_index, max_seq_length, short_seq_prob,
            vocab_words, rng))

  rng.shuffle(instances)
  return instances


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP]
  # Note than in LM, [CLS] is at the end of string (because attention constraint)
  max_num_tokens = max_seq_length - 2

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # In LM, we only have tokens_a
        tokens_a = []
        for j in range(len(current_chunk)):
          tokens_a.extend(current_chunk[j])

        truncate_seq(tokens_a, max_num_tokens, rng)

        assert len(tokens_a) >= 1

        tokens = []
        for token in tokens_a:
          tokens.append(token)

        tokens.append("[SEP]")

        tokens.append("[CLS]")

        (tokens, token_boundary) = create_lm_predictions(
             tokens, vocab_words, rng)
        instance = LMTrainingInstance(
            tokens=tokens,
            token_boundary=token_boundary,
            )
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


def _is_start_piece_sp(piece):
  """Check if the current word piece is the starting piece (sentence piece)."""
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  special_pieces.add(u"€".encode("utf-8"))
  special_pieces.add(u"£".encode("utf-8"))
  # Note(mingdachen):
  # For foreign characters, we always treat them as a whole piece.
  english_chars = set(list("abcdefghijklmnopqrstuvwxyz"))
  if (six.ensure_str(piece).startswith("▁") or
      six.ensure_str(piece).startswith("<") or piece in special_pieces or
      not all([str(i).lower() in english_chars.union(special_pieces)
               for i in piece])):
    return True
  else:
    return False


def _is_start_piece_bert(piece):
  """Check if the current word piece is the starting piece (BERT)."""
  # When a word has been split into
  # WordPieces, the first token does not have any marker and any subsequence
  # tokens are prefixed with ##. So whenever we see the ## token, we
  # append it to the previous set of word indexes.
  return not six.ensure_str(piece).startswith("##")


def is_start_piece(piece):
  if FLAGS.spm_model_file:
    return _is_start_piece_sp(piece)
  else:
    return _is_start_piece_bert(piece)


def create_lm_predictions(tokens, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  # Note(mingdachen): We create a list for recording if the piece is
  # the starting piece of current token, where 1 means true, so that
  # on-the-fly whole word masking is possible.
  token_boundary = [0] * len(tokens)

  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      token_boundary[i] = 1
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and not is_start_piece(token)):
      pass
    else:
      if is_start_piece(token):
        token_boundary[i] = 1

  output_tokens = list(tokens)

  return (output_tokens, token_boundary)


def truncate_seq(tokens_a, max_num_tokens, rng):
  """Truncates a sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  print('Create tokenizer')
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))
 
  print('Start reading input files')
  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length,
      FLAGS.short_seq_prob, 
      rng)

  print('Number of instance = ' + str(len(instances)))
  tf.logging.info("number of instances: %i", len(instances))

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  print('Writing output files')
  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
