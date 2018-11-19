from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

def load_dataset(filename):
    return load(open(filename, 'rb'))


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max([len(s.split()) for s in lines])

def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')

tokenizer = create_tokenizer(trainLines)

length = max_length(trainLines)

vocab_size = len(tokenizer.word_index) + 1

print('Max document length: %d ' % length)
print('Vocabulary size: %d' % vocab_size)

trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
print(trainX.shape, testX.shape)

model = load_model('model.h5')

loss, acc = model.evaluate([trainX, trainX, trainX], array(trainLabels), verbose=0)
print('Train Accuaracy: %f' % (acc*100))

loss, acc = model.evaluate([testX, testX, testX], array(testLabels), verbose=0)
print('Test Accuaracy: %f' % (acc*100))
