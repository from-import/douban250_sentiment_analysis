from Craw import crawl_douban_movie_reviews
import pandas as pd
import numpy as np
import jieba
from keras.preprocessing import sequence
from keras.models import load_model
from keras import initializers
import warnings
import pickle
warnings.filterwarnings('ignore')

n = 240  # 爬取第240部电影
m = 5  # 爬取前5条评论
data = crawl_douban_movie_reviews(n, m)

positive_reviews = []
negative_reviews = []

# 加载模型和初始化词典及maxlen
custom_objects = {
    'Orthogonal': initializers.Orthogonal
}

model = load_model('my_model.h5', custom_objects=custom_objects)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

dict_df = pd.read_pickle('dict.pkl')

with open('maxlen.pkl', 'rb') as f:
    maxlen = pickle.load(f)

def classify_review(review):
    # 分词函数
    cw = lambda x: list(jieba.cut(str(x)))

    test_data = [review]
    test_data = pd.DataFrame({'rateContent': test_data})
    test_data['words'] = test_data['rateContent'].apply(cw)

    # 打印分词结果
    print(f"分词结果: {test_data['words'].iloc[0]}")

    # 将分词后的数据转换为模型可以接受的格式
    get_sent = lambda x: [dict_df['id'][word] for word in x if word in dict_df['id']]
    test_data['sent'] = test_data['words'].apply(get_sent)

    # 打印词向量序列
    print(f"词向量序列: {test_data['sent'].iloc[0]}")

    test_data['sent'] = list(sequence.pad_sequences(test_data['sent'], maxlen=maxlen))

    # 打印填充后的序列
    print(f"填充后的序列: {test_data['sent'].iloc[0]}")

    # 使用模型进行预测
    prediction = model.predict(np.array(list(test_data['sent'])))

    # 打印预测结果
    for i in prediction:
        print(i[0])

    # 根据模型的预测输出，判断测试数据的情感倾向
    sentiment = ['positive' if i[0] > 0.5 else 'negative' for i in prediction]
    return sentiment

for movie_name, review in data:  # 解包data中的每一项，将电影名和评论分别赋值给movie_name和review
    sentiment = classify_review(review)[0]  # 获取评论的情感判断
    if sentiment == 'positive':
        positive_reviews.append(review)
    else:
        negative_reviews.append(review)

print("Positive Reviews:\n")
for review in positive_reviews:
    print(f"Review: {review}\n")

print("Negative Reviews:\n")
for review in negative_reviews:
    print(f"Review: {review}\n")
