from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers,initializers
import pandas as pd

problem = 3
encoding_dim = 5
input_dim = 16

print str(encoding_dim) + ' perceptrons'

df_encoding_data = pd.DataFrame()
df_decoding_data = pd.DataFrame()
# for i in range(5):
#   print "ITERATION: " + str(i)

#this creates an input vector shape
input_data = Input(shape=(input_dim,))
#this initializes the first layer squashing function
encoded = Dense(encoding_dim, activation='sigmoid', kernel_initializer=initializers.RandomUniform(), bias_initializer='zero')(input_data)
#this initializes the second layer squashing function
decoded = Dense(input_dim, activation='sigmoid', kernel_initializer=initializers.RandomUniform(), bias_initializer='zero')(encoded)

#creates model for input to representatation
autoencoder = Model(input_data, decoded)
#creates layer for input to hidden
encoder = Model(input_data, encoded)

#initializes the output vector from hidden layer
encoded_output = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_output, decoder_layer(encoded_output))

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

#train our autoencoder to learn some weights
autoencoder.fit(x_train,x_train,
epochs=200000,
verbose=2,
batch_size=256,
shuffle=True,
validation_data=(x_train, x_train))
print '3 perceptrons'

#this is the output of the hidden layer and the binary representation
encoded_data = encoder.predict(x_train)

df = pd.DataFrame(encoded_data)
df_encoding_data = pd.concat([df_encoding_data, df])


#this is the final output representation
decoded_data = decoder.predict(encoded_data)
df = pd.DataFrame(decoded_data)
df_decoding_data = pd.concat([df_decoding_data, df])


file_name = 'P' + str(problem) + '_encodedData.csv'
df_encoding_data.to_csv(file_name)
print "Finished Encoded to CSV"



file_name = 'P' + str(problem) + '_decodedData.csv'
df_decoding_data.to_csv(file_name)
print "Finished Decoded to CSV"














