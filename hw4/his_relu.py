from keras.layers import Input, Dense
from keras import optimizers,initializers
from keras.models import Model, Sequential
import pandas as pd
import time
import numpy as np


activation_func = 'sigmoid'

input_data = Input(shape=(16,))
print activation_func + ' AE'
encodedh = Dense(8, activation=activation_func,kernel_initializer=initializers.Constant(value=0.2),
                bias_initializer='zero')(input_data)
encoded = Dense(4, activation=activation_func,kernel_initializer=initializers.Constant(value=0.2),
                bias_initializer='zero')(encodedh)
decodedh = Dense(8, activation=activation_func,kernel_initializer=initializers.Constant(value=0.2),
                bias_initializer='zero')(encoded)
decoded = Dense(16, activation=activation_func,kernel_initializer=initializers.Constant(value=0.2),
                bias_initializer='zero')(decodedh)


autoencoder = Model(input_data, decoded)
encoderh = Model(input_data, encodedh)
encoder = Model(input_data, encoded)
decoderh = Model(input_data, decodedh)



# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

rms = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


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
                verbose=2,
                epochs=10000,
                batch_size=256,
                shuffle=True,
                validation_data=(x_train, x_train))


half_encoded = encoderh.predict(x_train)
print(half_encoded)
dfhalf_encoded = pd.DataFrame(half_encoded)
dfhalf_encoded.to_csv('half_encoded' + activation_func + '.csv')


encoded_data = encoder.predict(x_train)
print(encoded_data)
df_encoded = pd.DataFrame(encoded_data)
df_encoded.to_csv('encoded' + activation_func + '.csv')


half_decoded = decoderh.predict(x_train)
print(half_decoded)
dfhalf_decoded = pd.DataFrame(half_decoded)
dfhalf_decoded.to_csv('half_decoded' + activation_func + '.csv')

decoded_data = autoencoder.predict(x_train)
print(decoded_data)
df_decoded = pd.DataFrame(decoded_data)
df_decoded.to_csv('decoded' + activation_func + '.csv')