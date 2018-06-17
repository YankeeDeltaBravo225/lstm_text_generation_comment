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
import re


# 各文字とidとの相互変換を行うためのクラス
class CharTable:
    def __init__(self, text):
        uniq_chars = sorted(list(set(text)))
        self.id_num = len(uniq_chars)
        self.char2id_dict = dict((c, i) for i, c in enumerate(uniq_chars))
        self.id2char_dict = dict((i, c) for i, c in enumerate(uniq_chars))


    # 文字列をIDの列に変換
    def str2ids(self, str):
        return [ self.char2id_dict[char] for char in str]


    # ID列を文字列に変換
    def ids2str(self, ids):
        return ''.join([ self.id2char_dict[id] for id in ids])

    # IDを文字に変換
    def id2char(self, id):
        return self.id2char_dict[id]


    # 文字をIDに変換
    def char2id(self, char):
        return self.char2id_dict[id]


# テストデータのペア(現在の文字列、その次の文字)
class TrainPair:
    def __init__(self, text_ids, start_pos, len):
        last_pos = start_pos + len
        next_pos = last_pos + 1

        self.cur_char_ids = text_ids[start_pos : last_pos]
        self.next_char_id = text_ids[next_pos]


# ファイルをダウンロード
def download_file(url, filename):
    local_path = get_file(filename, origin=url)
    return local_path


# テキストを読み込んで、整形
def load_text(path):
    with io.open(path, encoding='utf-8') as f:
        raw_text = f.read()

    # 改行をすべて削除し、アルファベット大文字は小文字に変換
    text = re.sub(r'\n+', '', raw_text).lower()
    print('text length:', len(text))

    return text


# シーケンス文字列とその次に来る文字のリストを作成
def create_train_pair(text_ids, seq_len, skip):
    last_start_idx = len(text_ids) - seq_len - 1
    train_pairs = [TrainPair(text_ids, i, seq_len) for i in range(0, last_start_idx, skip)]

    print('number of sequences:', len(train_pairs))

    return train_pairs


# idx ⇒ char、char ⇒ idx のdictを作成
def create_char_dict(uniq_chars):
    dict_c2i = dict((c, i) for i, c in enumerate(uniq_chars))
    dict_i2c = dict((i, c) for i, c in enumerate(uniq_chars))

    return dict_c2i, dict_i2c


# 各シーケンス文字列・次文字を学習用のX, Yベクトルに変換
def vectorize_train_pairs(train_pairs, id_num):
    seq_num = len(train_pairs)
    seq_len = len(train_pairs[0].cur_char_ids)

    # 各シーケンス中に現れる各文字の出現確率を設定する配列
    x = np.zeros((seq_num, seq_len, id_num), dtype=np.bool)

    # 各シーケンスの次の文字の出現確率を設定する配列
    y = np.zeros((seq_num, id_num), dtype=np.bool)

    # vector化は各「文」について実施
    for seq_index, train_pair in enumerate(train_pairs):
        # 現在のシーケンス中の各文字の出現確率を1(100%)に設定
        for char_pos, cur_char_id in enumerate(train_pair.cur_char_ids):
            x[seq_index, char_pos, cur_char_id] = 1

        # 現在のシーケンスの次に来る文字種の出現確率を1(100%)に設定
        y[seq_index, train_pair.next_char_id] = 1

    return x, y


# build the model: a single LSTM
def create_model(seq_size, num_class):
    lstm = LSTM(
        units=128,
        input_shape=(seq_size, num_class)
    )

    model = Sequential()
    model.add(lstm)
    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=0.01)
    )
    return model


# 各「字」の出現確率の配列(ndarray型)から、出力する文字を選ぶ
# 単純に一番確率の高いものを選ぶのではなく、出現率に従いランダムに選ぶ
#
# predsはモデルからの出力であり、多項分布の形になっているため、
# その総和は必ず 1.0 となる
#
#  preds       : モデルからの出力結果、float32型の多項分布が入ったndarray
#  temperature : 多様度、この値が高いほど preds 中の出現率が低いものが選ばれやすくなる
def draw_lottery(preds, temperature=1.0):
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


# 現在の「文」の中のどの位置に何の「字」があるかのテーブルを
# フィッティング時に入力したxベクトルと同じフォーマットで生成
# 最初の次元は「文」のIDなので0固定
def create_pred_x(char_ids, char_id_num):
    seq_size = len(char_ids)
    x_pred = np.zeros((1, seq_size, char_id_num))
    for char_pos, char_id in enumerate(char_ids):
        x_pred[0, char_pos, char_id] = 1

    return x_pred


# Fitting終了後に呼ばれる文字列生成関数
# diversityとは多様性を意味する言葉
# この値が低いとモデルの予測で出現率が高いとされた「字」がそのまま選ばれ、
# 高ければそうでない「字」が選ばれる確率が高まる
def generate_text(model, seed_seq, char_table, diversity):
    print('----- Generating with diversity:%f' % (diversity))
    generated = ''
    seq_char_ids = seed_seq

    # 上記のランダムで選ばれた「文」に続く400個の「字」をモデルから予測し出力
    for i in range(400):
        x_pred = create_pred_x(seq_char_ids, char_table.id_num)

        # 予測により、各文字が現れる確率が多項分布で得られる
        y_pred = model.predict(x_pred, verbose=0)[0]

        # 確率にしたがって抽選を行う
        next_char_id = draw_lottery(y_pred, diversity)

        # 予測して得られた「字」を生成し、「文」に追加
        generated += char_table.id2char(next_char_id)

        # モデル入力するシーケンス文から最初の文字を削り、末尾に予測結果を追加
        seq_char_ids.pop(0)
        seq_char_ids.append(next_char_id)

    print(generated)


def main():
    # 設定
    seq_str_len = 40 #シーケンス1つのの文字数

    # ニーチェの文集をダウンロード or コマンドラインで入力テキストファイルを指定
    print('Loading the text file...')
    # text_path = download_file('https://s3.amazonaws.com/text-datasets/nietzsche.txt', 'nietzsche.txt')
    text_path = sys.argv[1]
    text      = load_text(text_path)

    print('Vectorization of text...')
    char_table = CharTable(text)
    text_ids   = char_table.str2ids(text)
    train_pairs = create_train_pair(text_ids, seq_str_len, 3)
    seed_char_ids = random.choice(train_pairs).cur_char_ids

    x_train, y_train = vectorize_train_pairs(train_pairs, char_table.id_num)

    print('Build model...')
    model = create_model(seq_str_len, char_table.id_num)

    print('Fit model...')
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=60
        )

    print('----- Generating text after fitting with seed "%s"' % (char_table.ids2str(seed_char_ids)) )
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        generate_text(model, seed_char_ids, char_table, diversity)

if __name__ == '__main__':
    main()
