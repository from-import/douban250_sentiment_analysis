from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np
import jieba
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# 加载正负样本
neg = pd.read_excel('neg.xls', header=None, index_col=None)
pos = pd.read_excel('pos.xls', header=None, index_col=None)

# 添加标签
pos['mark'] = 1
neg['mark'] = 0
pn = pd.concat([pos, neg], ignore_index=True)

# 分词
cw = lambda x: list(jieba.cut(str(x)))
pn['words'] = pn[0].apply(cw)

# 构建词典
all_words = []
for words in pn['words']:
    all_words.extend(words)

dict_df = pd.DataFrame(pd.Series(all_words).value_counts())
dict_df['id'] = list(range(1, len(dict_df) + 1))

# 词向量转换
def get_sent(words):
    return [dict_df['id'][word] for word in words if word in dict_df['id']]

pn['sent'] = pn['words'].apply(get_sent)

# 填充序列
maxlen = 50
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

# 数据准备
x = np.array(list(pn['sent']))[::2]
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2]
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent']))
ya = np.array(list(pn['mark']))

# 构建模型
print('Build model...')
model = Sequential()
model.add(Embedding(len(dict_df) + 1, 256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 保存词典和maxlen
dict_df.to_pickle('dict.pkl')
with open('maxlen.pkl', 'wb') as f:
    pickle.dump(maxlen, f)

# 训练模型
model.fit(x, y, batch_size=16, epochs=5)

# 评估模型
probs = model.predict(xt)
classes = (probs > 0.5).astype('int32').reshape(-1)
model.save('my_model.h5')

acc = accuracy_score(yt, classes)
print('Test accuracy:', acc)
