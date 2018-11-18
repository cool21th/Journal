from numpy import array

from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.utils import plot_model

seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))

seq_out = seq_in[:, 1: :]
n_out = n_in -1

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in, 1)))
model.add(RepeatVector(n_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')

model.fit(seq_in, seq_out, epochs=300, verbose=0)
yhat = model.predict(seq_in, verbose=0)
print(yhat[0,:,0])

