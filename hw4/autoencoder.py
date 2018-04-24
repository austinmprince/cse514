from keras.layers import Input, Dense
from keras import optimizers,initializers
from keras.models import Model
import pandas as pd
import numpy as np

initalVec = ['initializers.Zeros()','initializers.Ones()', 'initializers.constant(value=0.5)', 'initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)', 'initializers.constant(value=0.2)' ]

# for i in range(0, 5):
#
#     for j in range(3, 6):
#         print j

encoding_dim = 5


input_data = Input(shape=(16,))

encoded = Dense(encoding_dim, activation='sigmoid',kernel_initializer=initializers.RandomUniform(),
                bias_initializer='zero')(input_data)


decoded = Dense(16, activation='sigmoid',kernel_initializer=initializers.RandomUniform(),
                bias_initializer='zero')(encoded)
# encoded = Dense(encoding_dim, activation='sigmoid',kernel_initializer=eval(initalVec[i]),
#                 bias_initializer='zero')(input_data)
#
# decoded = Dense(16, activation='sigmoid',kernel_initializer=eval(initalVec[i]),
#                 bias_initializer='zero')(encoded)

autoencoder = Model(input_data, decoded)


encoder = Model(input_data, encoded)

encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))



# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# learning rate of 0.001 epsilon is the cutoff of the loss function
# after about 130,000 iterations this seems to converge to a loss value of around 1.2354e-07

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


autoencoder.fit(x_train, x_train,
                epochs=200000,
                verbose=2,
                batch_size=256,
                shuffle=True,
                validation_data=(x_train, x_train))


encoded_data = encoder.predict(x_train)
print(encoded_data)
# print encoded_data.shape
df = pd.DataFrame(encoded_data)


#
#
#
#
decoded_data = decoder.predict(encoded_data)
print(decoded_data)
df_out = pd.DataFrame(decoded_data)
print df
# df.to_csv('encodedData' + str(j) + 'Layer' + initalVec[i] + '.csv')
df.to_csv('encodedData3Layer1.csv')
print df_out
# df_out.to_csv('unEncoded' + str(j) + 'Layer' + initalVec[i] + '.csv')
# print decoded_data.shape
df_out.to_csv('decodedData3Layer1.csv')



