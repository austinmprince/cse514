from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import tempfile

import tempfile

dataPath = '/Users/macuser/Downloads/514hw1csv.csv'
CSV_COLUMNS = ['TIME', 'TEMP', 'HUM', 'LIGHT', 'CLOUD']

temp = tf.feature_column.numeric_column('temp')
num = tf.feature_column.numeric_column('hum')
light = tf.feature_column.numeric_column('light')
cloud = tf.feature_column.numeric_column('cloud')

base_columns = [ temp, num, light, cloud ]

def input_fn(data_file, num_epochs, shuffle, train):
# Input builder function
    if train is True:
        df_data = pd.read_csv(
          tf.gfile.Open(data_file),
          names=CSV_COLUMNS,
          nrows=5,
          skiprows=1
          )
    else:
        df_data = pd.read_csv(
          tf.gfile.Open(data_file),
          names=CSV_COLUMNS,
          skipinitialspace=True,
          engine="python",
          skiprows=5)
    # remove NaN elements
    df_data = df_data.dropna(how="any", axis=0)
    labels = df_data['TIME'].apply(lambda x: 0 if x > 0 else 1)
    return tf.estimator.inputs.pandas_input_fn(
          x=df_data,
          y=labels,
          batch_size=100,
          num_epochs=num_epochs,
          shuffle=shuffle,
          num_threads=5)

def build_estimator(model_dir, model_type):
  """Build an estimator."""
  if model_type == "wide":
    m = tf.estimator.LinearClassifier(
        model_dir=model_dir, feature_columns=base_columns)
  return m




df_data = pd.read_csv(
      tf.gfile.Open('editedHW1.csv'),names=CSV_COLUMNS,  engine="python", skiprows=1, nrows=5)

# df_train = pd.read_csv('editedHW1.csv', names=CSV_COLUMNS, skiprows=1, nrows=5)
# df_test = pd.read_csv('editedHW1.csv', names=CSV_COLUMNS,  skiprows=5)




# train_labels = df_data['TIME']
# # train_labels = df_train['TIME']
# test_labels = df_test['TIME']


model_dir = tempfile.mkdtemp()
m = tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=base_columns)

m.train(input_fn(dataPath, num_epochs=None, shuffle=True, train=True), steps=1000)

# train_input_fn = tf.estimator.inputs.pandas_input_fn(x=df_data, y=train_labels, shuffle=True)
#
# test_input_fn = tf.estimator.inputs.pandas_input_fn(x=df_test, y=test_labels, num_epochs=1, shuffle=False)

results = m.evaluate( input_fn=test_input_fn, steps=None)
print("model directory = %s" % model_dir)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))


