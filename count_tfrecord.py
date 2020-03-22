import sys
import tensorflow as tf

file_path =  str(sys.argv[1])
print('Reading: ' + file_path)

c = 0
for record in tf.python_io.tf_record_iterator(file_path):
  c += 1

print(str(c) + ' record(s).')
