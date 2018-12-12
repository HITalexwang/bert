# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random

import tokenization
import tensorflow as tf

class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_translate_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_translate_next = is_translate_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_translate_next: %s\n" % self.is_translate_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files,
                                    use_masked_lm=False):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if use_masked_lm:
      masked_lm_positions = list(instance.masked_lm_positions)
      masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
      masked_lm_weights = [1.0] * len(masked_lm_ids)

      while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_translate_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])
    if use_masked_lm:
      features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
      features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
      features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    
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
                              dupe_factor,  masked_lm_prob,
                              max_predictions_per_seq, rng, reverse_trans=False,
                              use_masked_lm=False):
  """Create `TrainingInstance`s from raw text."""
  sentences = []

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()
        line0, line1 = line.split('|||')
        tokens0 = tokenizer.tokenize(line0)
        tokens1 = tokenizer.tokenize(line1)
        if tokens0 and tokens1:
          sentences.append((tokens0, tokens1))

  vocab_words = list(tokenizer.vocab.keys())
  if reverse_trans:
    reverse_sentences = [(t[1], t[0]) for t in sentences]
  instances = []
  for _ in range(dupe_factor):
    for sentence_index in range(len(sentences)):
      instances.append(
          create_instance_from_sentences(
              sentences, sentence_index, max_seq_length, 
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng, use_masked_lm))
    if reverse_trans:
      for sentence_index in range(len(reverse_sentences)):
        instances.append(
          create_instance_from_sentences(
              reverse_sentences, sentence_index, max_seq_length, 
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng, use_masked_lm))

  rng.shuffle(instances)
  return instances


def create_instance_from_sentences(
    sentences, sentence_index, max_seq_length, 
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, use_masked_lm=False):
  """Creates `TrainingInstance`s for a single pair of paralleled sentence."""
  sentence = sentences[sentence_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  tokens_a = sentence[0]
  # Random next
  is_translate_next = False
  if rng.random() < 0.5:
    is_translate_next = True

    # This should rarely go for more than one iteration for large
    # corpora. However, just to be careful, we try to make sure that
    # the random document is not the same as the document
    # we're processing.
    for _ in range(10):
      random_sentence_index = rng.randint(0, len(sentences) - 1)
      if random_sentence_index != sentence_index:
        break

    random_sentence = sentences[random_sentence_index]
    tokens_b = random_sentence[1]
  # Actual next
  else:
    is_translate_next = False
    tokens_b = sentence[1]
  truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

  assert len(tokens_a) >= 1
  assert len(tokens_b) >= 1

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)

  tokens.append("[SEP]")
  segment_ids.append(0)

  for token in tokens_b:
    tokens.append(token)
    segment_ids.append(1)
  tokens.append("[SEP]")
  segment_ids.append(1)

  if use_masked_lm:
    (tokens, masked_lm_positions,
      masked_lm_labels) = create_masked_lm_predictions(
        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
  else:
    masked_lm_positions = None
    masked_lm_labels = None
  instance = TrainingInstance(
      tokens=tokens,
      segment_ids=segment_ids,
      is_translate_next=is_translate_next,
      masked_lm_positions=masked_lm_positions,
      masked_lm_labels=masked_lm_labels)

  return instance

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token

    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS. FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng, reverse_trans=FLAGS.reverse_trans, use_masked_lm=FLAGS.use_masked_lm)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


#if __name__ == "__main__":
#  flags.mark_flag_as_required("input_file")
#  flags.mark_flag_as_required("output_file")
#  flags.mark_flag_as_required("vocab_file")
#  tf.app.run()
