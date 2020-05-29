# %%

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%

import random
import os

import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

import json

with open('dic.json', 'r') as f:
    for line in f:
        dic = json.loads(line)
    ix2word, word2ix = dic['ix2word'], dic['word2ix']


class PoetryModel():
    def __init__(self):
        self.model = None
        self.do_train = True
        self.load_model = True
        self.poems = self.process_data()
        self.poems_num = len(self.poems)
        self.model_name = 'D:\python\dataset\\tang\\tang.hdf5'
        self.Embedding_dim = 128
        self.batch_size = 32
        self.epoch = 512
        self.max_len = 6
        if os.path.exists(self.model_name) and self.load_model:

            self.model = load_model(self.model_name)
            print('loaded')
        else:
            self.train()

    def process_data(self):
        files_content = []
        with open('poetry.txt', 'r', encoding='utf-8') as f:
            for line in f:
                x = line.strip()
                x = x.split(":")[1]
                if len(x) <= 5:
                    continue
                if x[5] == '，':
                    files_content.append(x)
        return files_content

    def build_model(self):
        '''建立模型'''
        print('building model')

        # 输入的dimension
        self.model = Sequential()
        self.model.add(Input(shape=(self.max_len, len(word2ix))))
        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(Dropout(0.6))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.6))
        self.model.add(Dense(len(ix2word), activation='softmax'))

        optimizer = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def generate_sample_result(self, epoch, logs):
        with open('out.txt', 'a', encoding='utf-8') as f:
            f.write('==================Epoch {}=====================\n'.format(epoch))

        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.7, 1.0, 1.3]:
            generate = self.predict_random(temperature=diversity)
            # 训练时的预测结果写入txt
            with open('out.txt', 'a', encoding='utf-8') as f:
                f.write(generate + '\n')

    def predict_random(self, temperature=1.0):
        '''随机从库中选取一句开头的诗句，生成五言绝句'''
        if not self.model:
            print('model not loaded')
            return

        index = random.randint(0, self.poems_num)
        sentence = self.poems[index][: self.max_len]
        generate = self.predict_sen(sentence, temperature=temperature)
        return generate

    def predict_first(self, char, temperature=1):
        '''根据给出的首个文字，生成五言绝句'''
        if not self.model:
            print('model not loaded')
            return

        index = random.randint(0, self.poems_num)
        # 选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
        sentence = self.poems[index][1 - self.max_len:] + char
        generate = str(char)
        #         print('first line = ',sentence)
        # 直接预测后面23个字符
        generate += self._preds(sentence, length=23, temperature=temperature)
        return generate

    def predict_sen(self, text, temperature=1.0):
        '''根据给出的前max_len个字，生成诗句'''
        '''此例中，即根据给出的第一句诗句（含逗号），来生成古诗'''
        if not self.model:
            return
        max_len = self.max_len
        if len(text) < max_len:
            print('length should not be less than ', max_len)
            return

        sentence = text[-max_len:]
        #         print('the first line:', sentence)
        generate = str(sentence)
        generate += self._preds(sentence, length=24 - max_len, temperature=temperature)
        return generate

    def predict_hide(self, text, single_len=5 ,temperature=1):
        '''根据给4个字，生成藏头诗五言绝句'''
        if not self.model:
            print('model not loaded')
            return
        if len(text) != 4:
            print('藏头诗的输入必须是4个字！')
            return

        index = random.randint(0, self.poems_num)
        # 选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
        sentence = self.poems[index][1 - self.max_len:] + text[0]
        generate = str(text[0])
        print('first line = ', sentence)

        for i in range(single_len):
            next_char = self._pred(sentence, temperature)
            sentence = sentence[1:] + next_char
            generate += next_char

        for i in range(3):
            generate += text[i + 1]
            sentence = sentence[1:] + text[i + 1]
            for i in range(single_len):
                next_char = self._pred(sentence, temperature)
                sentence = sentence[1:] + next_char
                generate += next_char

        return generate

    def sample(self, preds, temperature=1.0):
        '''
        当temperature=1.0时，模型输出正常
        当temperature=0.5时，模型输出比较open
        当temperature=1.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
        '''
        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds, 1. / temperature)
        preds = exp_preds / np.sum(exp_preds)
        pro = np.random.choice(range(len(preds)), 1, p=preds)
        return int(pro.squeeze())

    def _preds(self, sentence, length=23, temperature=1.0):
        '''
        sentence:预测输入值
        lenth:预测出的字符串长度
        供类内部调用，输入max_len长度字符串，返回length长度的预测值字符串
        '''
        sentence = sentence[:self.max_len]
        generate = ''
        for i in range(length):
            pred = self._pred(sentence, temperature)
            generate += pred
            sentence = sentence[1:] + pred
        return generate

    def _pred(self, sentence, temperature=.0):
        '''内部使用方法，根据一串输入，返回单个预测字符'''
        if len(sentence) < self.max_len:
            print('in def _pred,length error ')
            return

        sentence = sentence[-self.max_len:]
        x_pred = np.zeros(shape=(1, self.max_len, len(word2ix)))
        for index, char in enumerate(sentence):
            try:
                x_pred[0, index, word2ix[char]] = 1.0
            except:
                continue
        # x_pred = np.zeros((1, self.max_len, ))
        # for t, char in enumerate(sentence):
        #     x_pred[0, t, word2ix(char)] = 1.
        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds, temperature=temperature)
        next_char = ix2word[str(next_index)]

        return next_char

    def data_generator(self):
        '''生成器生成数据'''
        np.random.shuffle(self.poems)
        while 1:
            poem = np.random.choice(self.poems)
            # 选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
            try:
                for poem_index in range(len(poem) - self.max_len):
                    x = poem[poem_index: poem_index + self.max_len]
                    y = poem[poem_index + self.max_len]
                    # y_vec = np.zeros(
                    #     shape=(1, len(word2ix)),
                    #     dtype=np.bool
                    # )
                    # y_vec[0, word2ix[y]] = 1.0
                    y_vec = np.zeros(
                        shape=(1, len(word2ix)),
                        # dtype=np.bool
                    )
                    y_vec[0, word2ix[y]] = 1.0
                    x_vec = np.zeros(shape=(1, self.max_len, len(word2ix)))
                    for index, char in enumerate(x):
                        x_vec[0, index, word2ix[char]] = 1.0
                    # print(x_vec,y_vec)
                    yield x_vec, y_vec
            except:
                continue

    def train(self):
        '''训练模型'''
        print('training')
        number_of_epoch = self.poems_num

        if not self.model:
            self.build_model()

        self.history = self.model.fit(
            x=self.data_generator(),
            # verbose=True,
            steps_per_epoch=number_of_epoch // 10,
            epochs=self.epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.model_name, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )


model = PoetryModel()
print(model.predict_hide('爱我中华',single_len = 11))
# model.train()
# print(model.history.history)
#
# with open('history.txt', 'a') as outfile:
#     json.dump(model.history.history, outfile)

