import time
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers,initializers
import pandas as pd

input_dim = 16
layer_1_dim = 8
layer_2_dim = 4
layer_3_dim = 8
problem = 6

#for problem, activation_func in enumerate(['sigmoid', 'relu']):
activation_func = 'sigmoid'
df_encoding_data_1 = pd.DataFrame()
df_encoding_data_2 = pd.DataFrame()
df_encoding_data_3 = pd.DataFrame()
df_decoding_data = pd.DataFrame()

df_encoding_data_1_weights = pd.DataFrame()
df_encoding_data_2_weights = pd.DataFrame()
df_encoding_data_3_weights = pd.DataFrame()
df_decoding_weights = pd.DataFrame()

for i in range(5):
  print "FUNTION: " + activation_func + " & ITERATION: " + str(i)

  input_data = Input(shape=(input_dim,))

  layer_1_weights = Dense(layer_1_dim, activation=activation_func, kernel_initializer=initializers.RandomUniform(), bias_initializer='zero')(input_data)
  layer_2_weights = Dense(layer_2_dim, activation=activation_func, kernel_initializer=initializers.RandomUniform(), bias_initializer='zero')(layer_1_weights)
  layer_3_weights = Dense(layer_3_dim, activation=activation_func, kernel_initializer=initializers.RandomUniform(), bias_initializer='zero')(layer_2_weights)
  decoded = Dense(input_dim, activation=activation_func, kernel_initializer=initializers.RandomUniform(), bias_initializer='zero')(layer_3_weights)

  autoencoder = Model(input_data, decoded)

  #layer 1
  encoder_1 = Model(input_data, layer_1_weights)

  #layer 2
  encoded_1_input = Input(shape=(layer_1_dim,))
  layer = autoencoder.layers[2]
  encoder_2 = Model(encoded_1_input, layer(encoded_1_input))

  #layer 3
  encoded_2_input = Input(shape=(layer_2_dim,))
  layer = autoencoder.layers[3]
  encoder_3 = Model(encoded_2_input, layer(encoded_2_input))

  #layer 4
  encoded_3_input = Input(shape=(layer_3_dim,))
  layer = autoencoder.layers[4]
  decoder = Model(encoded_3_input, layer(encoded_3_input))

  rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
  autoencoder.compile(optimizer=rms, loss='binary_crossentropy')

  x_train = [[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]]

  start_time = time.time()
  autoencoder.fit(x_train,x_train,
    epochs=200000,
    verbose=2,
    batch_size=256,
    shuffle=True,
    validation_data=(x_train, x_train))
  print("--- %s seconds ---" % (time.time() - start_time))
  #save encoder 1
  encoded_data_1 = encoder_1.predict(x_train)
  df = pd.DataFrame(encoded_data_1)
  df_encoding_data_1 = pd.concat([df_encoding_data_1, df])

  df_weights = pd.DataFrame(encoder_1.get_weights())
  df_encoding_data_1_weights = pd.concat([df_encoding_data_1_weights, df_weights])

  #save encoder 2
  encoded_data_2 = encoder_2.predict(encoded_data_1)
  df = pd.DataFrame(encoded_data_2)
  df_encoding_data_2 = pd.concat([df_encoding_data_2, df])

  df_weights = pd.DataFrame(encoder_2.get_weights())
  df_encoding_data_2_weights = pd.concat([df_encoding_data_2_weights, df_weights])

  #save encoder 3
  encoded_data_3 = encoder_3.predict(encoded_data_2)
  df = pd.DataFrame(encoded_data_3)
  df_encoding_data_3 = pd.concat([df_encoding_data_3, df])

  df_weights = pd.DataFrame(encoder_3.get_weights())
  df_encoding_data_3_weights = pd.concat([df_encoding_data_3_weights, df_weights])

  #save encoder 4
  decoder_data = decoder.predict(encoded_data_3)
  df = pd.DataFrame(decoder_data)
  df_decoding_data = pd.concat([df_decoding_data, df])

  df_weights = pd.DataFrame(decoder.get_weights())
  df_decoding_weights = pd.concat([df_decoding_weights, df_weights])

  file_name = "P" + str(problem + 4) + activation_func + "_encoded_1.csv"
  df_encoding_data_1.to_csv(file_name)

  # file_name = "P" + str(problem + 4) + "_encoded_1_weights.csv"
  # df_encoding_data_1_weights.to_csv(file_name)

  file_name = "P" + str(problem + 4) + activation_func + "_encoded_2.csv"
  df_encoding_data_2.to_csv(file_name)

  # file_name = "P" + str(problem + 4) + "_encoded_2_weights.csv"
  # df_encoding_data_2_weights.to_csv(file_name)

  file_name = "P" + str(problem + 4) + activation_func + "_encoded_3.csv"
  df_encoding_data_3.to_csv(file_name)

  # file_name = "P" + str(problem + 4) + "_encoded_3_weights.csv"
  # df_encoding_data_3_weights.to_csv(file_name)

  file_name = "P" + str(problem + 4) + activation_func +"_decodedData.csv"
  df_decoding_data.to_csv(file_name)

  # file_name = "P" + str(problem + 4) + "_decoded_weights.csv"
  # df_decoding_weights.to_csv(file_name)

print("DONE")

