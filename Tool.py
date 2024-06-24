# coding=utf-8
from __future__ import absolute_import, print_function
from keras.models import load_model
import pandas as pd
import numpy as np
import jieba
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from sklearn.metrics import accuracy_score
import pickle
from keras import initializers
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

def Classification(test_data):
    # 自定义初始化器
    custom_objects = {
        'Orthogonal': initializers.Orthogonal
    }

    model = load_model('my_model.h5', custom_objects=custom_objects)

    # 显式编译模型，消除警告
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 加载词典和maxlen
    dict = pd.read_pickle('dict.pkl')

    with open('maxlen.pkl', 'rb') as f:
        maxlen = pickle.load(f)

    # 分词函数
    cw = lambda x: list(jieba.cut(str(x)))

    test_data = [test_data]
    test_data = pd.DataFrame({'rateContent': test_data})
    test_data['words'] = test_data['rateContent'].apply(cw)

    # 将分词后的数据转换为模型可以接受的格式
    get_sent = lambda x: [dict.get(word, 0) for word in x]  # 使用 dict.get 而不是 dict[word]，如果 word 不存在就返回 0
    test_data['sent'] = test_data['words'].apply(get_sent)
    test_data['sent'] = list(sequence.pad_sequences(test_data['sent'], maxlen=maxlen))

    # 使用模型进行预测
    prediction = model.predict(np.array(list(test_data['sent'])))

    # 根据模型的预测输出，判断测试数据的情感倾向
    sentiment = ['positive' if i[0] > 0.5 else 'negative' for i in prediction]
    return sentiment

