from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from pickle import dump

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()

    file.close()

    return text

def clean_doc(doc):
    tokens = doc.split()

    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens

def process_docs(directory, is_train):
    documents = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue

        if not is_train and not filename.startswith('cv9'):
            continue

        path = directory + '/' + filename

        doc = load_doc(path)

        tokens = clean_doc(doc)

        documents.append(tokens)

    return documents

def save_dataset(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)


negative_docs = process_docs('review_polarity/txt_sentoken/neg', True)
positive_docs = process_docs('review_polarity/txt_sentoken/pos', True)
trainX = negative_docs + positive_docs
trainy = [0 for _ in range(900)] + [1 for _ in range(900)]

save_dataset([trainX, trainy], 'train.pkl')

negative_docs = process_docs('review_polarity/txt_sentoken/neg', False)
negative_docs = process_docs('review_polarity/txt_sentoken/neg', False)
testX = negative_docs + positive_docs
testy = [0 for _ in range(100)] + [1 for _ in range(100)]
save_dataset([testX, testy], 'test.pkl')

