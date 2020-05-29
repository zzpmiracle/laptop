from gensim.models import Word2Vec
import gensim
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import Input

data_base_path = '/kaggle/input'

seq_length = 50
EPOCH = 20
batch_size = 32


def build_w2v():
    model_path = os.path.join(data_base_path, 'wiki_word2vec_50.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model


w2v_model = build_w2v


def get_data(mode='train'):
    data_path = os.path.join(data_base_path, mode + '.txt')
    corpus = []
    with open(data_path, encoding='utf-8') as f:
        for line in f.readlines():
            if line:
                corpus.append(line)
    return corpus


train_corpus, val_corpus, test_corpus = get_data('train'), get_data('validation'), get_data('test')


def data_generator(corpus):
    i = 0
    x_tmp, y_tmp = np.zeros(shape=(batch_size, seq_length, w2v_model.vector_size)), np.zeros(shape=(batch_size))
    while (1):
        np.random.shuffle(corpus)
        for line in corpus:
            line = line.split()
            try:
                y = line[0]
                index = 0
                for _, word in enumerate(line[1:]):
                    if index >= seq_length:
                        break
                    if word not in w2v_model:
                        continue
                    x_tmp[0, index] = w2v_model[word]
                    index += 1
                y_tmp[i] = int(y)
                i += 1
                if i % batch_size == 0:
                    yield x_tmp, y_tmp
            except IOError:
                continue


# def data_generator():
#     np.random.shuffle(train_data)
#     for line in train_data:
#         line = line.split()
#         try:
#             x_tmp, y_tmp= np.zeros(shape=(1,seq_length,len(word2ix))), np.zeros(shape=(1,2))
#             y = line[0]
#             for index, word in enumerate(line[1:]):
#                 if index >= seq_length:
#                     break
#                 if word not in word2ix:
#                     continue
#                 x_tmp[0,index, word2ix[word]] = 1.0
#             y_tmp[0,int(y)] = 1.0
#             yield x_tmp, y_tmp
#         except IOError:
#             print(line)
#             continue

# val_x, val_y = get_data(mode='validation')
# print(len(val_x))
# test_x, text_y = get_data(mode='test')
def creat_model():
    model = Sequential()
    model.add(Input(shape=(seq_length, len(word2ix))))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.6))
    model.add(LSTM(units=64))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


model = creat_model()

history = model.fit(x=data_generator(train_corpus), steps_per_epoch=len(train_corpus) // batch_size, epochs=EPOCH)

model_path = 'senti_embedding.hdf5'
model.save(model_path)
print(history.history)
