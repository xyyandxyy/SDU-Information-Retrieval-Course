import gensim
import json


def deal_Tweet():
    setences = []
    all_txt = []
    with open("data/Tweet/Tweet", 'r', encoding='utf-8') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        for id in all_lines:
            data_dic = json.loads(id)
            all_txt.append(data_dic["text"])
    for setence in all_txt:
        setences.append(setence.split())
    model = gensim.models.Word2Vec(setences, min_count=1, size=48)
    model.wv.save_word2vec_format('./data/Tweet/model_tweet.txt', binary=False)
    print("Tweet has done!")


def deal_News():
    setences = []
    all_txt = []
    with open("data/NEWS/News", 'r', encoding='utf-8') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        for id in all_lines:
            data_dic = json.loads(id)
            all_txt.append(data_dic["text"])
    for setence in all_txt:
        setences.append(setence.split())
    model = gensim.models.Word2Vec(setences, min_count=1, size=48)
    model.wv.save_word2vec_format('./data/NEWS/model_News.txt', binary=False)
    print("News has done!")


def deal_20ng():
    setences = []
    all_txt = []
    with open("data/20ng/20ng", 'r', encoding='utf-8') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        for id in all_lines:
            data_dic = json.loads(id)
            all_txt.append(data_dic["text"])
    for setence in all_txt:
        setences.append(setence.split())
    model = gensim.models.Word2Vec(setences, min_count=1, size=48)
    model.wv.save_word2vec_format('./data/20ng/model_20ng.txt', binary=False)
    print("20ng has done!")


deal_Tweet()
deal_20ng()
deal_News()
