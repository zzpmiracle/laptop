from gensim.models import Word2Vec
import gensim
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM,Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

data_base_path = 'D:\python\dataset\sentiment'

seq_length = 80
EPOCH = 10
batch_size = 128
Embedding_dim = 100
# only one start
def build_word2id():
    """
    :param file: word2id保存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = [data_base_path+'\\train.txt', data_base_path+'\\validation.txt']
    print(path)
    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    # with open(file, 'w', encoding='utf-8') as f:
    #     for w in word2id:
    #         f.write(w+'\t')
    #         f.write(str(word2id[w]))
    #         f.write('\n')
    return word2id


# word2id, length = build_word2id()
# print(length)
# word_vecs = build_word2vec(word2id)
#
# w2v_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
# vec_length = w2v_model.vector_size
word2id = build_word2id()

def get_data(mode='train'):
    data_path = os.path.join(data_base_path,mode+'.txt')
    x, y=[], []
    blank_vec = 0
    with open(data_path,encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                try:
                    x_tmp, y_tmp= [], sp[0]
                except:
                    # print(line)
                    pass
                for word in sp[1:]:
                    try:
                        x_tmp.append(word2id[word])
                    except KeyError:
                        pass
                if len(x_tmp) >= seq_length:
                    x_tmp = x_tmp[:seq_length]
                else:
                    x_tmp.extend([blank_vec for _ in range(seq_length-len(x_tmp))])
                x.append(x_tmp)
                y.append(y_tmp)
            y = to_categorical(y)
    return np.array(x),np.array(y)


train_x, train_y = get_data(mode='train')
val_x, val_y = get_data(mode='validation')
test_x, text_y = get_data(mode='test')
print(train_x.shape,train_y.shape)
def creat_model():
    model = Sequential()
    model.add(Embedding(input_dim=len(word2id),output_dim=Embedding_dim))
    model.add(LSTM(units=32,input_shape=[seq_length,Embedding_dim], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    return model

model = creat_model()

model.fit(train_x,train_y,validation_data=[val_x,val_y],epochs=EPOCH,batch_size=batch_size,verbose=2)

model_path = 'senti_embedding.hdf5'
model.save(model_path)




