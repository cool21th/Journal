from numpy import array
from keras.models import Model
from keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from keras.utils import plot_model

seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))

seq_out = seq_in[:, 1:, :]
n_out = n_in -1

visible = Input(shape=(n_in, 1))
encoder = LSTM(100, activation='relu')(visible)

decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)

decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)

model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')
plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')

model.fit(seq_in, [seq_in, seq_out], epochs=300, verbose=0)

yhat = model.predict(seq_in, verbose=0)
print(yhat)
