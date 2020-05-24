from gensim.models import Word2Vec
import gensim
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow import keras
data_base_path = 'D:\python\dataset\sentiment'

model_path = os.path.join(data_base_path,'wiki_word2vec_50.bin')
seq_length = 80
EPOCH = 10
batch_size = 16
# only one start
def build_word2id(file='./w2v'):
    """
    :param file: word2id保存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = [data_base_path+'\\train.txt', data_base_path+'\\validation.txt']
    print(path)
    length = {i:0 for i in range(1000)}
    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                length[len(sp)]+=1
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    # with open(file, 'w', encoding='utf-8') as f:
    #     for w in word2id:
    #         f.write(w+'\t')
    #         f.write(str(word2id[w]))
    #         f.write('\n')
    return word2id,length


def build_word2vec(word2id, fname=model_path, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    import gensim
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs

# word2id, length = build_word2id()
# print(length)
# word_vecs = build_word2vec(word2id)

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
vec_length = w2v_model.vector_size
def get_data(mode='train'):
    data_path = os.path.join(data_base_path,mode+'.txt')
    x, y=[], []
    blank_vec = [0 for _ in range(vec_length)]
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
                        x_tmp.append(w2v_model[word])
                    except KeyError:
                        pass
                if len(x_tmp) >= seq_length:
                    x_tmp = x_tmp[:seq_length]
                else:
                    x_tmp.extend([blank_vec for _ in range(seq_length-len(x_tmp))])
                x.append(x_tmp)
                y.append(y_tmp)
            y = keras.utils.to_categorical(y)
    return np.array(x),np.array(y)


train_x, train_y = get_data(mode='train')
val_x, val_y = get_data(mode='validation')
test_x, text_y = get_data(mode='test')
print(train_x.shape,train_y.shape)
def creat_model():
    model = Sequential()
    model.add(LSTM(units=32,input_shape=[seq_length,vec_length], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    return model

model = creat_model()

model.fit(train_x,train_y,validation_data=[val_x,val_y],epochs=EPOCH,batch_size=batch_size,verbose=2)




