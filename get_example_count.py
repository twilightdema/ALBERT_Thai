import sys
import tensorflow as tf
import squad_utils
import classifier_utils

processor = classifier_utils.ColaProcessor(use_spm=True, do_lower_case=True)
examples = processor.get_train_examples('glue_data')
print('CoLA Train: ' + str(len(examples)))

processor = classifier_utils.MnliProcessor(use_spm=True, do_lower_case=True)
examples = processor.get_train_examples('glue_data')
print('MNLI Train: ' + str(len(examples)))

processor = classifier_utils.MrpcProcessor(use_spm=True, do_lower_case=True)
examples = processor.get_train_examples('glue_data')
print('MRPC Train: ' + str(len(examples)))

processor = classifier_utils.QnliProcessor(use_spm=True, do_lower_case=True)
examples = processor.get_train_examples('glue_data')
print('QNLI Train: ' + str(len(examples)))

processor = classifier_utils.QqpProcessor(use_spm=True, do_lower_case=True)
examples = processor.get_train_examples('glue_data')
print('QQP Train: ' + str(len(examples)))

processor = classifier_utils.RteProcessor(use_spm=True, do_lower_case=True)
examples = processor.get_train_examples('glue_data')
print('RTE Train: ' + str(len(examples)))

processor = classifier_utils.Sst2Processor(use_spm=True, do_lower_case=True)
examples = processor.get_train_examples('glue_data')
print('SST-2 Train: ' + str(len(examples)))

processor = classifier_utils.StsbProcessor(use_spm=True, do_lower_case=True)
examples = processor.get_train_examples('glue_data')
print('STS-B Train: ' + str(len(examples)))

examples = squad_utils.read_squad_examples(input_file='squad_data/train-v1.1.json', is_training=True)
print('SQuAD Train: ' + str(len(examples)))

examples = squad_utils.read_squad_examples(input_file='squad_data/train-v2.0.json', is_training=True)
print('SQuADV2 Train: ' + str(len(examples)))
