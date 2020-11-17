import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import auxilary

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import Normalizer


import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras import regularizers
from keras.models import load_model

extract_heartbeat = True

if extract_heartbeat:
    train_filename = 'all_heartbeats_eh_filtered.npy'
    encoder_filename = 'encoder_eh.h5'
    autoencoder_filename = 'model_autoencoder_eh.h5'
else:
    train_filename = 'all_heartbeats.npy'
    encoder_filename = 'encoder.h5'
    autoencoder_filename = 'model_autoencoder.h5'




X_train = np.load(train_filename)
normalizer = Normalizer()

input_dim = X_train.shape[1]
encoding_dim = 120

# Autoencoder
autoencoder = Sequential()
# Encoded Layers
autoencoder.add(
    Dense(encoding_dim, input_shape=(input_dim,), activation='relu')
)

# autoencoder.add(
#     Dense(6*encoding_dim, input_shape=(input_dim,), activation='relu')
# )
# autoencoder.add(
#     Dense(encoding_dim, activation='relu')
# )

# Decoder Layers
# autoencoder.add(
#     Dense(6*encoding_dim, activation='relu')
# )
# autoencoder.add(
#     Dense(12*encoding_dim, activation='relu')
# )
autoencoder.add(
    Dense(input_dim, activation='sigmoid')
)

autoencoder.summary()

# Get the Encoder Model
input_signal = Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
#encoder_layer2 = autoencoder.layers[1]
#encoder_layer3 = autoencoder.layers[2]
encoder = Model(input_signal, encoder_layer1(input_signal))
encoder.summary()

# Save the encoder
encoder.save(encoder_filename)
print("Encoder is saved")

# Compile and fit the model
X_train, X_val, y_train, y_val = train_test_split(X_train, X_train, test_size = 0.1, random_state = 42, shuffle = True)

X_train = normalizer.fit_transform(X_train)
noise = np.random.normal(0, 0.01, (X_train.shape))
print('shape of noise: ', noise.shape)
X_train = X_train + noise
X_val = normalizer.transform(X_val)

print("shape of train after split:", X_train.shape)
print('shape of test after split: ', X_val.shape)

opt = keras.optimizers.Adam(lr = 0.001)

autoencoder.compile(optimizer=opt, loss='mse')
autoencoder.fit(X_train, X_train, epochs=30, batch_size=512, validation_data=(X_val, X_val) )

autoencoder.save(autoencoder_filename)

# Test the model
test_model = load_model(autoencoder_filename)
loss = test_model.evaluate(X_val, X_val)
print('loss on val is ', loss)

# Test the encoder
test_encoder = load_model(encoder_filename)
patient = X_train[0,:].reshape(1, -1)
decoded_template = encoder.predict(patient)
print("shape of decoded: ", decoded_template.shape)
print(decoded_template)