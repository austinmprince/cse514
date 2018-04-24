from keras.layers import Input, Dense
from keras import optimizers,initializers
from keras.models import Model
import pandas as pd
import time
import sys
import numpy as np

initalVec = ['initializers.Zeros()','initializers.Ones()', 'initializers.constant(value=0.5)', 'initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)', 'initializers.constant(value=0.2)' ]
timeList = []
for i in range(0, 5):
#
#     for j in range(3, 6):
#         print j

    print 'iteration ' + str(i)
    print 'relu activation function'
    t = time.time()

    input_data = Input(shape=(16,))

    encoded = Dense(8, activation='relu',kernel_initializer=initializers.RandomUniform(),
                    bias_initializer='zero')(input_data)

    encoded_half = Dense(4, activation='relu',kernel_initializer=initializers.RandomUniform(),
                    bias_initializer='zero')(encoded)

    decoded_half = Dense(8, activation='relu',kernel_initializer=initializers.RandomUniform(),
                    bias_initializer='zero')(encoded_half)


    decoded = Dense(16, activation='relu',kernel_initializer=initializers.RandomUniform(),
                    bias_initializer='zero')(decoded_half)
    #

    autoencoder = Model(input_data, decoded)

    encoder = Model(input_data, encoded)


    encoder_half_input = Input(shape=(8,))
    encoder_half = Model(encoder_half_input, autoencoder.layers[2](encoder_half_input))
    decoder_half_input = Input(shape=(4,))
    decoder_half = Model(decoder_half_input, autoencoder.layers[3](decoder_half_input))
    decoder_input = Input(shape=(8,))
    decoder = Model(decoder_input, autoencoder.layers[4](decoder_input))


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
                    epochs=10,
                    verbose=2,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_train, x_train))

    print 'elapsed time' + str(t - time.time())
    timeList.append(t - time.time())
    print 'relu activation function'

    encoded_data = encoder.predict(x_train)
    print(encoded_data)
    # print encoded_data.shape
    df_encoded = pd.DataFrame(encoded_data)
    df_encoded.to_csv('encodedrelu' + str(i) + '.csv')

    encoded_data_half = encoder_half.predict(encoded_data)
    df_half_enc = pd.DataFrame(encoded_data_half)
    print df_half_enc
    df_half_enc.to_csv('half_encrelu' + str(i) + '.csv')

    decoded_data_half = decoder_half.predict(encoded_data_half)
    df_half_dec = pd.DataFrame(decoded_data_half)
    df_half_dec.to_csv('half_decrelu' + str(i) + '.csv')
    print df_half_dec



#
#
#
#
    decoded_data = autoencoder.predict(decoded_data_half)
    print(decoded_data)
    df_out = pd.DataFrame(decoded_data)
    df_out.to_csv('dec_datarelu' + str(i) + '.csv')

# df.to_csv('encodedData' + str(j) + 'Layer' + initalVec[i] + '.csv')
# df.to_csv('encodedData3Layer1.csv')
# print df_out
# # df_out.to_csv('unEncoded' + str(j) + 'Layer' + initalVec[i] + '.csv')
# # print decoded_data.shape
# df_out.to_csv('decodedData3Layer1.csv')
print timeList



