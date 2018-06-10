'''
Original file : https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

# ニーチェの文集をダウンロードする
# path : ダウンロードした先のパス
path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

# text : 入力ファイル
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

# chars : 重複を排除した「字」のリスト
chars = sorted(list(set(text)))
print('total chars:', len(chars))

# char_indices : 「字」を上記charsのindex番号に変換するdict
char_indices = dict((c, i) for i, c in enumerate(chars))

# indices_char : 上記と逆にindex番号を「字」に変換するdict
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
# maxlen : いくつの「字」を1つの「文」とするか
maxlen = 40

# step : 開始位置のスキップ数
step = 3

# sentences  : 「文」のリスト
sentences = []

# next_chars : 各「文」について、その次の「文」の最初の「字」
next_chars = []

for i in range(0, len(text) - maxlen, step):
    # 単純に長さで区切った部分文字列を一つの文という扱いで抽出
    sentences.append(text[i: i + maxlen])

    # 次の文の最初の文字を保存
    next_chars.append(text[i + maxlen])

# 上記の「文」の数をそのままLSTMのsequence数として用いる
print('nb sequences:', len(sentences))

print('Vectorization...')

# x : np.bool型 3次元配列 [文の数, 文の最大長, 字の種類]　⇒ 文中の各位置に各indexの文字が出現するか
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)

# y : np.bool型 2次元配列 [文の数, 字の種類]              ⇒ 次の文の開始文字のindex
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

# vector化は各「文」について実施
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# ここら辺はただのsingle LSTMのため説明は省略

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# 勾配法にRMSpropを用いる
# 以下参照
# https://qiita.com/tokkuman/items/1944c00415d129ca0ee9

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# 各「字」の出現確率の配列(ndarray型)から、出力する文字を選ぶ
# 単純に一番確率の高いものを選ぶのではなく、出現率に従いランダムに選ぶ
#
# predsはモデルからの出力であり、多項分布の形になっているため、
# その総和は必ず 1.0 となる
#
#  preds       : モデルからの出力結果、float32型の多項分布が入ったndarray
#  temperature : 多様度、この値が高いほど preds 中の出現率が低いものが選ばれやすくなる
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array

    # 64bit float型に変換
    preds = np.asarray(preds).astype('float64')

    # 確率の低く出た「字」が抽選で選ばれやすくなるようにゲタをはかせるため、
    # 自然対数を取った上、引数の値で割る
    # 参照
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.log.html
    preds = np.log(preds) / temperature

    # 上記で確率の自然対数を取ったため、その逆変換である自然指数関数をとる
    # 参照
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.exp.html
    exp_preds = np.exp(preds)

    # 多項分布の形に合わせるため、総和が1となるように全値を総和で割る
    preds = exp_preds / np.sum(exp_preds)

    # 多項分布に基づいた抽選を行う
    # 参照
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 各epoch終了時に呼ばれるcallback
def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    # モデルは40文字の「文」からその次の「字」を予測するものであるため、
    # その元となる40文字の「文」を入力テキストからランダムに選ぶ
    start_index = random.randint(0, len(text) - maxlen - 1)

    # diversityとは多様性を意味する言葉
    # この値が低いとモデルの予測で出現率が高いとされた「字」がそのまま選ばれ、
    # 高ければそうでない「字」が選ばれる確率が高まる
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''

        # 元にする「文」を選択
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        # 上記のランダムで選ばれた「文」に続く400個の「字」をモデルから予測し出力する
        for i in range(400):

            # 現在の「文」の中のどの位置に何の「字」があるかのテーブルを
            # フィッティング時に入力したxベクトルと同じフォーマットで生成
            # 最初の次元は「文」のIDなので0固定
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            # 現在の「文」に続く「字」を予測する
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            # 予測して得られた「字」を生成し、「文」に追加
            generated += next_char

            # モデル入力する「文」から最初の文字を削り、予測結果の「字」を追加
            # 例：sentence 「これはドイツ製」
            #     next_char 「の」
            #     ↓
            #     sentence 「れはドイツ製の」
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# 各epoch終了時のcallbackとして、上記のon_epoch_endを呼ぶ
# 参照
# https://keras.io/ja/callbacks/#lambdacallback
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# フィッティング実施、各epoch完了時に先述の on_epoch_end が呼ばれる
model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])