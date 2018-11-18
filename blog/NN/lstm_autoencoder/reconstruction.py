from numpy import array
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.utils import plot_model

sequence = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

model.fit(sequence, sequence, epochs=300, verbose=0)

model = Model(inputs=model.inputs, output=model.layers[0].output)
plot_model(model, show_shapes=True, to_file='lstm_encoder.png')
yhat = model.predict(sequence)
print(yhat.shape)
print(yhat)
