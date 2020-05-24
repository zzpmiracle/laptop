import random
import os

import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model,Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense,Embedding
from tensorflow.keras.optimizers import Adam

test = np.load('tang.npz')
data, ix2word, word2ix = test['data'], test['ix2word'].item(), test['word2ix'].item()


class PoetryModel():
    def __init__(self):
        self.model = None
        self.do_train = True
        self.load_model = False
        self.poems = self.process_data()
        self.poems_num = len(self.poems)
        self.model_name = 'tang.hdf5'
        self.Embedding_dim = 128
        self.batch_size = 32
        self.epoch = 100
        self.max_len = 6
        if os.path.exists(self.model_name) and self.load_model:
            self.model = load_model(self.model_name)
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
        self.model.add(Embedding(input_dim=len(word2ix), output_dim=self.Embedding_dim))
        self.model.add(LSTM(512, input_shape=[self.max_len,self.Embedding_dim],return_sequences=True))
        self.model.add(Dropout(0.6))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.6))
        self.model.add(Dense(len(ix2word),activation='softmax'))
        # input_tensor = Input(shape=(self.max_len, len(word2ix)))
        # lstm = (input_tensor)
        # dropout = Dropout(0.6)(lstm)
        # lstm = LSTM(256)(dropout)
        # dropout = Dropout(0.6)(lstm)
        # dense = Dense(len(word2ix), activation='softmax')(dropout)
        # self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def generate_sample_result(self, epoch, logs):
        '''训练过程中，每4个epoch打印出当前的学习情况'''
        if epoch % 4 != 0:
            return

        with open('out/out.txt', 'a', encoding='utf-8') as f:
            f.write('==================Epoch {}=====================\n'.format(epoch))

        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.7, 1.0, 1.3]:
            print("------------Diversity {}--------------".format(diversity))
            generate = self.predict_random(temperature=diversity)
            print(generate)

            # 训练时的预测结果写入txt
            with open('out/out.txt', 'a', encoding='utf-8') as f:
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
        print('the first line:', sentence)
        generate = str(sentence)
        generate += self._preds(sentence, length=24 - max_len, temperature=temperature)
        return generate

    def predict_hide(self, text, temperature=1):
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

        for i in range(5):
            next_char = self._pred(sentence, temperature)
            sentence = sentence[1:] + next_char
            generate += next_char

        for i in range(3):
            generate += text[i + 1]
            sentence = sentence[1:] + text[i + 1]
            for i in range(5):
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
        x_pred = []
        for char in sentence:
            x_pred.append(word2ix[char])
        # x_pred = np.zeros((1, self.max_len, ))
        # for t, char in enumerate(sentence):
        #     x_pred[0, t, word2ix(char)] = 1.
        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds, temperature=temperature)
        next_char = ix2word[next_index]

        return next_char

    def data_generator(self):
        '''生成器生成数据'''
        np.random.shuffle(self.poems)
        index = 0
        while 1:
            # 选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
            poem = self.poems[index]
            try:
                for poem_index in range(len(poem)-self.max_len):
                    x = poem[poem_index: poem_index + self.max_len]
                    y = poem[poem_index + self.max_len]
                    # y_vec = np.zeros(
                    #     shape=(1, len(word2ix)),
                    #     dtype=np.bool
                    # )
                    # y_vec[0, word2ix[y]] = 1.0
                    y_vec = np.zeros(
                        shape=(1,len(word2ix)),
                        # dtype=np.bool
                    )
                    y_vec[0,word2ix[y]] = 1.0
                    x_vec = []
                    for char in x:
                        x_vec.append(word2ix[char])
                    # print(x_vec,y_vec)
                    yield np.array(x_vec), y_vec
                index +=1
            except:
                continue

    def train(self):
        '''训练模型'''
        print('training')
        number_of_epoch = self.poems_num//50
        print('epoches = ', number_of_epoch)
        print('poems_num = ', self.poems_num)
        # print('len(self.files_content) = ', len(self.files_content))

        if not self.model:
            self.build_model()

        self.model.fit(
            x=self.data_generator(),
            # verbose=True,
            steps_per_epoch=number_of_epoch,
            epochs=self.epoch,
            batch_size = self.batch_size,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.model_name, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )
model = PoetryModel()
model.train()