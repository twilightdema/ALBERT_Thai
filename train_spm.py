import os
import sys
import sentencepiece as spm

max_dict_size = 15000
input_raw_textfile = sys.argv[1]
output_modelfile = input_raw_textfile[0: input_raw_textfile.rfind('.')] + '.spm_model'
output_vocabfile = input_raw_textfile[0: input_raw_textfile.rfind('.')] + '.spm_vocab'

untokenized_input_path = input_raw_textfile[0: input_raw_textfile.rfind('.')] + '.untoken.txt'

# THWIKI and BEST2010 input data has already tokenized, we need to reconstruct untokenned text in order to train SPM model
count = 0
with open(untokenized_input_path, 'w', encoding='utf-8') as fout:
  with open(input_raw_textfile, 'r', encoding='utf-8') as fin:
    for line in fin:
      line = ''.join(line.split(' '))
      fout.write(line + '\n')
      # ALBERT need another \n to mark end of document
      fout.write('\n')
      print('Write document #' + str(count))
      count = count + 1

# Train sentence piece model
spm.SentencePieceTrainer.Train('--pad_id=0 --unk_id=1 --pad_piece=<pad> --unk_piece=<unk> --bos_id=-1 --eos_id=-1 --control_symbols=[CLS],[SEP],[MASK],<pad> --input=' + 
  untokenized_input_path + 
  ' --model_prefix=sp --vocab_size=' + str(max_dict_size) + ' --hard_vocab_limit=false')

#  --model_prefix=prefix_name --pad_id=0 --unk_id=1 --pad_piece=<pad> --unk_piece=<unk> --bos_id=-1 --eos_id=-1 --control_symbols=[CLS],[SEP],[MASK],<pad> --user_defined_symbols=(,),",-,.,–,£,€'
'''
spm.SentencePieceTrainer.Train('--pad_id=0 --bos_id=2 --eos_id=3 --unk_id=1 --user_defined_symbols=<MASK> --input=' + 
  local_untokened_data_file + 
  ' --model_prefix=sp --vocab_size=' + str(max_dict_size))
'''
# Move sp.model / sp.vocab to the dict paths
os.rename("sp.model", output_modelfile)
os.rename("sp.vocab", output_vocabfile)
