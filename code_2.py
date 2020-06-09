# 加入平仄规则
import random
import sys
import os
import re
import numpy as np

from tensorflow import keras
# import plaidml.keras
# plaidml.keras.install_backend()
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# import keras

from pypinyin import lazy_pinyin,Style
import warnings
warnings.filterwarnings('ignore')

'''
1   仄仄平平仄，平平仄仄平。平平平仄仄，仄仄仄平平。仄仄平平仄，平平仄仄平。平平平仄仄，仄仄仄平平。
2   仄仄仄平平，平平仄仄平。平平平仄仄，仄仄仄平平。仄仄平平仄，平平仄仄平。平平平仄仄，仄仄仄平平。
3   平平平仄仄，仄仄仄平平。仄仄平平仄，平平仄仄平。平平平仄仄，仄仄仄平平。仄仄平平仄，平平仄仄平。
4   平平仄仄平，仄仄仄平平。仄仄平平仄，平平仄仄平。平平平仄仄，仄仄仄平平。仄仄平平仄，平平仄仄平。

1   平平仄仄仄平平，仄仄平平仄仄平。仄仄平平平仄仄，平平仄仄仄平平。平平仄仄平平仄，仄仄平平仄仄平。仄仄平平平仄仄，平平仄仄仄平平。
2   平平仄仄平平仄，仄仄平平仄仄平。仄仄平平平仄仄，平平仄仄仄平平。平平仄仄平平仄，仄仄平平仄仄平。仄仄平平平仄仄，平平仄仄仄平平。
3   仄仄平平仄仄平，平平仄仄仄平平。平平仄仄平平仄，仄仄平平仄仄平。仄仄平平平仄仄，平平仄仄仄平平。平平仄仄平平仄，仄仄平平仄仄平。
4   仄仄平平平仄仄，平平仄仄仄平平。平平仄仄平平仄，仄仄平平仄仄平。仄仄平平平仄仄，平平仄仄仄平平。平平仄仄平平仄，仄仄平平仄仄平。
'''

poem_4_5 = [['zzppz', 'ppzzp', 'pppzz', 'zzzpp'], ['zzppz', 'ppzzp', 'pppzz', 'zzzpp'],
            ['zzzpp', 'ppzzp', 'pppzz', 'zzzpp'], ['zzppz', 'ppzzp', 'pppzz', 'zzzpp'],
            ['pppzz', 'zzzpp', 'zzppz', 'ppzzp'], ['pppzz', 'zzzpp', 'zzppz', 'ppzzp'],
            ['ppzzp', 'zzzpp', 'zzppz', 'ppzzp'], ['pppzz', 'zzzpp', 'zzppz', 'ppzzp']]
poem_8_5 = [['zzppz', 'ppzzp', 'pppzz', 'zzzpp', 'zzppz', 'ppzzp', 'pppzz', 'zzzpp'],
            ['zzzpp', 'ppzzp', 'pppzz', 'zzzpp', 'zzppz', 'ppzzp', 'pppzz', 'zzzpp'],
            ['pppzz', 'zzzpp', 'zzppz', 'ppzzp', 'pppzz', 'zzzpp', 'zzppz', 'ppzzp'],
            ['ppzzp', 'zzzpp', 'zzppz', 'ppzzp', 'pppzz', 'zzzpp', 'zzppz', 'ppzzp']]
poem_4_7 = [['ppzzzpp', 'zzppzzp', 'zzpppzz', 'ppzzzpp'], ['ppzzppz', 'zzppzzp', 'zzpppzz', 'ppzzzpp'],
            ['ppzzppz', 'zzppzzp', 'zzpppzz', 'ppzzzpp'], ['ppzzppz', 'zzppzzp', 'zzpppzz', 'ppzzzpp'],
            ['zzppzzp', 'ppzzzpp', 'ppzzppz', 'zzppzzp'], ['zzpppzz', 'ppzzzpp', 'ppzzppz', 'zzppzzp'],
            ['zzpppzz', 'ppzzzpp', 'ppzzppz', 'zzppzzp'], ['zzpppzz', 'ppzzzpp', 'ppzzppz', 'zzppzzp']]
poem_8_7 = [['ppzzzpp', 'zzppzzp', 'zzpppzz', 'ppzzzpp', 'ppzzppz', 'zzppzzp', 'zzpppzz', 'ppzzzpp'],
            ['ppzzppz', 'zzppzzp', 'zzpppzz', 'ppzzzpp', 'ppzzppz', 'zzppzzp', 'zzpppzz', 'ppzzzpp'],
            ['zzppzzp', 'ppzzzpp', 'ppzzppz', 'zzppzzp', 'zzpppzz', 'ppzzzpp', 'ppzzppz', 'zzppzzp'],
            ['zzpppzz', 'ppzzzpp', 'ppzzppz', 'zzppzzp', 'zzpppzz', 'ppzzzpp', 'ppzzppz', 'zzppzzp']]

poem_pingze_dict = {
    'poem_4_5': ['zzppzppzzppppzzzzzpp', 'zzppzppzzppppzzzzzpp', 'zzzppppzzppppzzzzzpp', 'zzppzppzzppppzzzzzpp',
                 'pppzzzzzppzzppzppzzp', 'pppzzzzzppzzppzppzzp', 'ppzzpzzzppzzppzppzzp', 'pppzzzzzppzzppzppzzp'],
    'poem_8_5': ['zzppzppzzppppzzzzzppzzppzppzzppppzzzzzpp', 'zzzppppzzppppzzzzzppzzppzppzzppppzzzzzpp',
                 'pppzzzzzppzzppzppzzppppzzzzzppzzppzppzzp', 'ppzzpzzzppzzppzppzzppppzzzzzppzzppzppzzp'],
    'poem_4_7': ['ppzzzppzzppzzpzzpppzzppzzzpp', 'ppzzppzzzppzzpzzpppzzppzzzpp', 'ppzzppzzzppzzpzzpppzzppzzzpp', 'ppzzppzzzppzzpzzpppzzppzzzpp',
                 'zzppzzpppzzzppppzzppzzzppzzp', 'zzpppzzppzzzppppzzppzzzppzzp', 'zzpppzzppzzzppppzzppzzzppzzp', 'zzpppzzppzzzppppzzppzzzppzzp'],
    'poem_8_7': ['ppzzzppzzppzzpzzpppzzppzzzppppzzppzzzppzzpzzpppzzppzzzpp', 'ppzzppzzzppzzpzzpppzzppzzzppppzzppzzzppzzpzzpppzzppzzzpp',
                 'zzppzzpppzzzppppzzppzzzppzzpzzpppzzppzzzppppzzppzzzppzzp', 'zzpppzzppzzzppppzzppzzzppzzpzzpppzzppzzzppppzzppzzzppzzp']
}


def get_pingze(x):
    pingze_dict = {0: '*', 1: 'p', 2: 'p', 3: 'z', 4: 'z'}
    shengdiao = lazy_pinyin(x, style=Style.TONE3)[0][-1]
    if shengdiao in ['1', '2', '3', '4']:
        return pingze_dict[int(shengdiao)]
    else:
        return '*'

class Config_5(object):
    poetry_file = 'poetry4min.txt'
    weight_file = 'poetry_model_4.h5'
    max_len = 4
    batch_size = 128
    learning_rate = 0.0005

class Config_7(object):
    poetry_file = 'poetry6min.txt'
    weight_file = 'poetry_model_6.h5'
    max_len = 6
    batch_size = 128
    learning_rate = 0.0005


def preprocess_file(Config):
    # 语料文本内容
    files_content = ''
    with open(Config.poetry_file, 'r', encoding='UTF-8') as f:
        for line in f:
            x = line.strip() + "]"
            # x = x.split(":")[1]
            if len(x) <= Config.max_len + 1:
                continue
            if x[Config.max_len + 1] == '，':
                files_content += x

    # 删除标点
    files_content = files_content.replace('，', '').replace('。', '').replace('？', '').replace('！', '')

    words = sorted(list(files_content))
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    Word_Pairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words, _ = zip(*Word_Pairs)
    # word到id的映射
    word2num = dict((c, i) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    Word_2numF = lambda x: word2num.get(x, len(words) - 1)
    num2pingze = dict((i, get_pingze(c)) for i, c in enumerate(words))
    return Word_2numF, num2word, words, files_content, num2pingze


class PoetryModel(object):
    def __init__(self, config):
        self.model = None
        self.config = config

        # 文件预处理
        self.word2numF, self.num2word, self.words, self.files_content, self.num2pingze = preprocess_file(self.config)

        # 诗的list
        self.poems = self.files_content.split(']')
        # 诗的总数量
        self.poems_num = len(self.poems)

        # 如果模型文件存在则直接加载模型，否则开始训练
        if os.path.exists(self.config.weight_file):
            self.model = keras.models.load_model(self.config.weight_file)
            keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True, to_file='poetry_model_' + str(self.config.max_len) + '.png')
            self.model.summary()
        else:
            self.train()

    def build_model(self):
        '''建立模型'''
        # 输入的dimension
        # input_tensor = keras.layers.Input(shape=(self.config.max_len,))
        # embedd = keras.layers.Embedding(len(self.num2word) + 2, 300, input_length=self.config.max_len)(input_tensor)
        # lstm = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(embedd)
        # # dropout = Dropout(0.6)(lstm)
        # # lstm = LSTM(256)(dropout)
        # # dropout = Dropout(0.6)(lstm)
        # flatten = keras.layers.Flatten()(lstm)
        # dense = keras.layers.Dense(len(self.words), activation=keras.activations.softmax)(flatten)
        # self.model = keras.models.Model(inputs=input_tensor, outputs=dense)
        # self.model.compile(loss=keras.losses.categorical_crossentropy,
        #                    optimizer=keras.optimizers.Adam(lr=self.config.learning_rate), metrics=['accuracy'])

        model = keras.models.Sequential()
        # model.add(keras.layers.Input(shape=(self.config.max_len,)))
        model.add(keras.layers.Embedding(input_dim=len(self.num2word) + 1, output_dim=300, input_length=self.config.max_len))
        model.add(keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True)))
        # model.add(keras.layers.Dropout(0.6))
        # model.add(keras.layers.LSTM(256))
        # model.add(keras.layers.Dropout(0.6))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(len(self.words), activation=keras.activations.softmax))
        self.model = model
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(lr=self.config.learning_rate), metrics=['accuracy'])

    def sample(self, preds, pingzeflag=True, pingze='p'):
        preds = np.asarray(preds).astype('float64')

        if pingzeflag:
            for idx, _ in enumerate(preds):
                if self.num2pingze[idx] != pingze:
                    preds[idx] = 0

        return np.argmax(preds)

    def data_generator(self):
        '''生成器生成数据'''
        i = 0
        while 1:
            try:
                x = self.files_content[i: i + self.config.max_len]
                y = self.files_content[i + self.config.max_len]
            except:
                i = 0
                x = self.files_content[i: i + self.config.max_len]
                y = self.files_content[i + self.config.max_len]

            if ']' in x or ']' in y:
                i += 1
                continue

            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
            y_vec[0, self.word2numF(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.config.max_len),
                dtype=np.int32
            )

            for t, char in enumerate(x):
                x_vec[0, t] = self.word2numF(char)
            yield x_vec, y_vec
            # i += 1
            # i += self.config.max_len + 1
            # print(x, y)

    def train(self):
        '''训练模型'''
        # number_of_epoch = len(self.files_content) / (self.config.max_len + 1) // self.config.batch_size
        number_of_epoch = len(self.files_content) // self.config.batch_size
        # number_of_epoch = len(self.files_content) - (self.config.max_len + 1) * self.poems_num
        # number_of_epoch /= self.config.batch_size
        # number_of_epoch = int(number_of_epoch / 1.5)

        if not self.model:
            self.build_model()

        self.model.summary()

        history = self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=5 * number_of_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False)
            ]
        )

        acc = history.history['acc']
        loss = history.history['loss']

        with open('modelResults.txt', 'a+') as f:
            f.write('acc\n')
            for item in acc:
                f.write("{}\n".format(item))
            f.write('loss\n')
            for item in loss:
                f.write("{}\n".format(item))

    def predict_hide(self, text, type=5, num_of_sentence=4):
        if not self.model:
            print('model not loaded')
            return

        index = random.randint(0, self.poems_num)
        # 选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
        sentence = self.poems[index][1 - self.config.max_len:] + text[0]
        generate = str(text[0])
        # print('first line = ', sentence)

        poem_pingze = poem_pingze_dict['poem_' + sys.argv[2] + '_' + sys.argv[3]]

        candpingze = []
        sentence_pingze = ''
        for sen in text:
            sentence_pingze += get_pingze(sen)

        for poem_pingze_i in poem_pingze:
            flag = 0
            for i in range(num_of_sentence):
                flag += int(sentence_pingze[i] == poem_pingze_i[type * i])
            if flag == num_of_sentence:
                candpingze.append(poem_pingze_i)


        pingzeflag = bool(len(candpingze))

        if pingzeflag:
            poem_pingze_cand = random.choice(candpingze)
            if type == 5:
                poem_pingze_cands = re.findall(r'.{5}', poem_pingze_cand)
            elif type == 7:
                poem_pingze_cands = re.findall(r'.{7}', poem_pingze_cand)

        for i in range(type - 1):
            if pingzeflag:
                next_char = self._pred(sentence, pingzeflag, pingze=poem_pingze_cands[0][i + 1])
            else:
                next_char = self._pred(sentence, pingzeflag)
            sentence = sentence[1:] + next_char
            generate += next_char

        for i in range(num_of_sentence - 1):
            generate += text[i + 1]
            sentence = sentence[1:] + text[i + 1]
            for j in range(type - 1):
                if pingzeflag:
                    next_char = self._pred(sentence, pingzeflag, pingze=poem_pingze_cands[i + 1][j + 1])
                else:
                    next_char = self._pred(sentence, pingzeflag)
                sentence = sentence[1:] + next_char
                generate += next_char

        if num_of_sentence == 4:
            return generate[:type] + '，\n' + generate[type: type * 2] + '。\n' + \
                   generate[type * 2: type * 3] + '，\n' + generate[type * 3: type * 4] + '。'
        elif num_of_sentence == 8:
            return generate[:type] + '，\n' + generate[type: type * 2] + '。\n' + \
                   generate[type * 2: type * 3] + '，\n' + generate[type * 3: type * 4] + '。\n' + \
                   generate[type * 4: type * 5] + '，\n' + generate[type * 5: type * 6] + '。\n' + \
                   generate[type * 6: type * 7] + '，\n' + generate[type * 7: type * 8] + '。'

    def predict_first(self, char, type=5, num_of_sentence=4):
        if not self.model:
            print('model not loaded')
            return

        index = random.randint(0, self.poems_num)
        # 选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
        sentence = self.poems[index][1 - self.config.max_len:] + char
        generate = str(char)

        generate += self._preds(sentence, length=type * num_of_sentence - 1)

        if num_of_sentence == 4:
            return generate[:type] + '，\n' + generate[type: type * 2] + '。\n' + \
                   generate[type * 2: type * 3] + '，\n' + generate[type * 3: type * 4] + '。'
        elif num_of_sentence == 8:
            return generate[:type] + '，\n' + generate[type: type * 2] + '。\n' + \
                   generate[type * 2: type * 3] + '，\n' + generate[type * 3: type * 4] + '。\n' + \
                   generate[type * 4: type * 5] + '，\n' + generate[type * 5: type * 6] + '。\n' + \
                   generate[type * 6: type * 7] + '，\n' + generate[type * 7: type * 8] + '。'


    def predict_sen(self, text, type=5, num_of_sentence=4):
        if not self.model:
            return
        max_len = self.config.max_len + 1
        if len(text) < max_len:
            print('length should not be less than ', max_len)
            return

        sentence = text[-max_len:]
        # print('the first line:', sentence)
        generate = str(sentence)

        generate += self._preds(sentence, length=type * num_of_sentence - max_len)

        if num_of_sentence == 4:
            return generate[:type] + '，\n' + generate[type: type * 2] + '。\n' + \
                   generate[type * 2: type * 3] + '，\n' + generate[type * 3: type * 4] + '。'
        elif num_of_sentence == 8:
            return generate[:type] + '，\n' + generate[type: type * 2] + '。\n' + \
                   generate[type * 2: type * 3] + '，\n' + generate[type * 3: type * 4] + '。\n' + \
                   generate[type * 4: type * 5] + '，\n' + generate[type * 5: type * 6] + '。\n' + \
                   generate[type * 6: type * 7] + '，\n' + generate[type * 7: type * 8] + '。'


    def _preds(self, sentence, length=19):
        """
        sentence:预测输入值
        length:预测出的字符串长度
        供类内部调用，输入max_len长度字符串，返回length长度的预测值字符串
        """
        # sentence = sentence[:self.config.max_len]
        generate = ''

        pingzeflag = True

        poem_pingze = poem_pingze_dict['poem_' + sys.argv[2] + '_' + sys.argv[3]]
        if sys.argv[1] == 'first':
            candpingze = []
            sentence_pingze = get_pingze(sentence[-1])

            for poem_pingze_i in poem_pingze:
                if sentence_pingze == poem_pingze_i[0]:
                    candpingze.append(poem_pingze_i)

            poem_pingze_cand = random.choice(candpingze)
            poem_pingze_cand = poem_pingze_cand[-length:]

        elif sys.argv[1] == 'sen':
            candpingze = []
            sentence_pingze = ''
            for sen in sentence:
                sentence_pingze += get_pingze(sen)

            for poem_pingze_i in poem_pingze:
                if sentence_pingze == poem_pingze_i[:len(sentence)]:
                    candpingze.append(poem_pingze_i)

            pingzeflag = bool(len(candpingze))
            if pingzeflag:
                poem_pingze_cand = random.choice(candpingze)
                poem_pingze_cand = poem_pingze_cand[-length:]

        for i in range(length):
            if pingzeflag:
                pred = self._pred(sentence, pingzeflag, pingze=poem_pingze_cand[i])
            else:
                pred = self._pred(sentence, pingzeflag)
            generate += pred
            sentence = sentence[1:] + pred
        return generate


    def _pred(self, sentence, pingzeflag=True, pingze='p'):
        """
        内部使用方法，根据一串输入，返回单个预测字符
        """
        if len(sentence) < self.config.max_len:
            print('in def _pred,length error ')
            return

        sentence = sentence[-self.config.max_len:]
        x_pred = np.zeros((1, self.config.max_len))
        for t, char in enumerate(sentence):
            x_pred[0, t] = self.word2numF(char)

        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds, pingzeflag=pingzeflag, pingze=pingze)
        next_char = self.num2word[next_index]

        return next_char


if __name__ == '__main__':
    # argv[1]   argv[2]   argv[3]     argv[4]                   description
    # hide      4 / 8     5 / 7       测试一下                   藏头诗
    # first     4 / 8     5 / 7       山                        根据给出的首个文字，生成诗
    # sen       4 / 8     5 / 7       春风起秋云 / 春风不知何处月   根据给出的前max_len + 1个字，生成诗句
    model_5 = PoetryModel(Config_5())
    model_7 = PoetryModel(Config_7())
    if sys.argv[1] == 'hide':
        if sys.argv[2] == '4':
            if sys.argv[3] == '5':
                print(model_5.predict_hide(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), text=sys.argv[4]))
            elif sys.argv[3] == '7':
                print(model_7.predict_hide(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), text=sys.argv[4]))
        elif sys.argv[2] == '8':
            if sys.argv[3] == '5':
                print(model_5.predict_hide(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), text=sys.argv[4]))
            elif sys.argv[3] == '7':
                print(model_7.predict_hide(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), text=sys.argv[4]))
    elif sys.argv[1] == 'first':
        if sys.argv[2] == '4':
            if sys.argv[3] == '5':
                print(model_5.predict_first(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), char=sys.argv[4]))
            elif sys.argv[3] == '7':
                print(model_7.predict_first(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), char=sys.argv[4]))
        elif sys.argv[2] == '8':
            if sys.argv[3] == '5':
                print(model_5.predict_first(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), char=sys.argv[4]))
            elif sys.argv[3] == '7':
                print(model_7.predict_first(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), char=sys.argv[4]))
    elif sys.argv[1] == 'sen':
        if sys.argv[2] == '4':
            if sys.argv[3] == '5':
                print(model_5.predict_sen(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), text=sys.argv[4]))
            elif sys.argv[3] == '7':
                print(model_7.predict_sen(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), text=sys.argv[4]))
        elif sys.argv[2] == '8':
            if sys.argv[3] == '5':
                print(model_5.predict_sen(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), text=sys.argv[4]))
            elif sys.argv[3] == '7':
                print(model_7.predict_sen(num_of_sentence=int(sys.argv[2]), type=int(sys.argv[3]), text=sys.argv[4]))

