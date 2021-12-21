# -*- coding: utf-8 -*-

import os
from collections import Counter
import json
import nltk
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from bert_serving.client import BertClient

def load_News(data_path='data/NEWS/',type='word2v'):
    wordinx = ""
    indx = ""
    vec=''
    if type=='bert':
        all_txt = []
        all_lable = []
        with open(data_path + 'News', 'r',encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            for id in all_lines:
                data_dic = json.loads(id)
                all_txt.append(data_dic["text"])
                all_lable.append(data_dic["cluster"])
            all_lines=all_txt
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())

            all_vector_representation = np.zeros(shape=(11107, 1024))
            bc = BertClient()
            for i, line in enumerate(all_lines):
                if(i%500==0):
                    print("embedding step: ", i,'/',len(all_lines))
                result = bc.encode([line])
                all_vector_representation[i] = result.squeeze()
        print('data have loaded')
        XX = all_vector_representation

        # with open(data_path + 'label_StackOverflow.txt',encoding='utf-8') as label_file:
        #     y = np.array(list((map(int, label_file.readlines()))))
        #     print(y.dtype)
        return XX, all_lable
    else:
        if type=='word2v':
            wordinx ='model_News_48_wordinx'
            indx='model_News_48_inx'
            vec='model_News_48_vec'
        if type=='wiki':
            wordinx = 'Wikipedia_index.dic'
            indx = 'vocab_emb_Word2vec_48_index.dic'
            vec = 'vocab_emb_Word2vec_48.vec'

        with open(data_path + wordinx, 'r', encoding='utf-8') as inp_indx, \
                open(data_path + indx, 'r') as inp_dic, \
                open(data_path + vec) as inp_vec:
            pair_dic = inp_indx.readlines()
            word_index = {}
            for pair in pair_dic:
                word, index = pair.replace('\n', '').split('\t')
                word_index[word] = index
            print(index)
            index_word = {v: k for k, v in word_index.items()}

            del pair_dic

            emb_index = inp_dic.readlines()
            emb_vec = inp_vec.readlines()
            word_vectors = {}
            for index, vec in zip(emb_index, emb_vec):
                word = index_word[index.replace('\n', '')]
                word_vectors[word] = np.array(list((map(float, vec.split()))))

            del emb_index
            del emb_vec
        all_txt = []
        all_lable = []
        with open(data_path + 'News', 'r', encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            for id in all_lines:
                data_dic = json.loads(id)
                all_txt.append(data_dic["text"])
                all_lable.append(data_dic["cluster"])
            all_lines = all_txt
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())
            unigram = {}
            for item in word_count.items():
                unigram[item[0]] = item[1] / total_count

            all_vector_representation = np.zeros(shape=(11107, 48))
            for i, line in enumerate(all_lines):
                word_sentence = nltk.word_tokenize(line)

                sent_rep = np.zeros(shape=[48, ])
                j = 0
                for word in word_sentence:
                    try:
                        wv = word_vectors[word]
                        j = j + 1
                    except KeyError:
                        continue

                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                if j != 0:
                    all_vector_representation[i] = sent_rep / j
                else:
                    all_vector_representation[i] = sent_rep

        pca = PCA(n_components=1)
        pca.fit(all_vector_representation)
        pca = pca.components_

        XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

        XX = XX1

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)

        # with open(data_path + 'label_StackOverflow.txt',encoding='utf-8') as label_file:
        #     y = np.array(list((map(int, label_file.readlines()))))
        #     print(y.dtype)
        print('data have loaded')
        return XX, all_lable
def load_tweet(data_path='data/Tweet/',type='word2v'):
    if type=='bert':
        all_txt = []
        all_lable = []
        with open(data_path + 'Tweet', 'r',encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            for id in all_lines:
                data_dic = json.loads(id)
                all_txt.append(data_dic["text"])
                all_lable.append(data_dic["cluster"])
            all_lines=all_txt
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())

            all_vector_representation = np.zeros(shape=(2471, 1024))
            bc = BertClient()
            for i, line in enumerate(all_lines):
                if(i%500==0):
                    print("embedding step: ", i,'/',len(all_lines))
                result = bc.encode([line])
                all_vector_representation[i] = result.squeeze()
        print('data have loaded')
        XX = all_vector_representation

        return XX, all_lable
    else:
        if type == 'word2v':
            wordinx = 'model_Tweet_48_wordinx'
            indx = 'model_Tweet_48_inx'
            vec = 'model_Tweet_48_vec'
        if type == 'wiki':
            wordinx = 'Wikipedia_index.dic'
            indx = 'vocab_emb_Word2vec_48_index.dic'
            vec = 'vocab_emb_Word2vec_48.vec'
        with open(data_path + wordinx, 'r', encoding='utf-8') as inp_indx, \
                open(data_path + indx, 'r') as inp_dic, \
                open(data_path + vec) as inp_vec:
            pair_dic = inp_indx.readlines()
            word_index = {}
            for pair in pair_dic:
                word, index = pair.replace('\n', '').split('\t')
                word_index[word] = index
            print(index)
            index_word = {v: k for k, v in word_index.items()}

            del pair_dic

            emb_index = inp_dic.readlines()
            emb_vec = inp_vec.readlines()
            word_vectors = {}
            for index, vec in zip(emb_index, emb_vec):
                word = index_word[index.replace('\n', '')]
                word_vectors[word] = np.array(list((map(float, vec.split()))))

            del emb_index
            del emb_vec
        all_txt = []
        all_lable = []
        with open(data_path + 'Tweet', 'r', encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            for id in all_lines:
                data_dic = json.loads(id)
                all_txt.append(data_dic["text"])
                all_lable.append(data_dic["cluster"])
            all_lines = all_txt
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())
            unigram = {}
            for item in word_count.items():
                unigram[item[0]] = item[1] / total_count

            all_vector_representation = np.zeros(shape=(2471, 48))
            for i, line in enumerate(all_lines):
                word_sentence = nltk.word_tokenize(line)

                sent_rep = np.zeros(shape=[48, ])
                j = 0
                for word in word_sentence:
                    try:
                        wv = word_vectors[word]
                        j = j + 1
                    except KeyError:
                        continue

                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                if j != 0:
                    all_vector_representation[i] = sent_rep / j
                else:
                    all_vector_representation[i] = sent_rep

        pca = PCA(n_components=1)
        pca.fit(all_vector_representation)
        pca = pca.components_

        XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

        XX = XX1

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)

        # with open(data_path + 'label_StackOverflow.txt',encoding='utf-8') as label_file:
        #     y = np.array(list((map(int, label_file.readlines()))))
        #     print(y.dtype)
        print('data have loaded')
        return XX, all_lable

def load_20ng(data_path='data/20ng/',type='word2v'):
    if type=='bert':
        all_txt = []
        all_lable = []
        with open(data_path + '20ng', 'r',encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            for id in all_lines:
                data_dic = json.loads(id)
                all_txt.append(data_dic["text"])
                all_lable.append(data_dic["cluster"])
            all_lines=all_txt
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())

            all_vector_representation = np.zeros(shape=(18845, 1024))
            bc = BertClient()
            for i, line in enumerate(all_lines):
                if(i%500==0):
                    print("embedding step: ", i,'/',len(all_lines))
                result = bc.encode([line])
                all_vector_representation[i] = result.squeeze()
        print('data have loaded')
        XX = all_vector_representation

        # scaler = MinMaxScaler()
        # XX = scaler.fit_transform(XX)

        # with open(data_path + 'label_StackOverflow.txt',encoding='utf-8') as label_file:
        #     y = np.array(list((map(int, label_file.readlines()))))
        #     print(y.dtype)
        return XX, all_lable
    else:
        if type == 'word2v':
            wordinx = 'model_20ng_48_wordinx'
            indx = 'model_20ng_48_inx'
            vec = 'model_20ng_48_vec'
        if type == 'wiki':
            wordinx = 'Wikipedia_index.dic'
            indx = 'vocab_emb_Word2vec_48_index.dic'
            vec = 'vocab_emb_Word2vec_48.vec'
        with open(data_path + wordinx, 'r', encoding='utf-8') as inp_indx, \
                open(data_path + indx, 'r') as inp_dic, \
                open(data_path + vec) as inp_vec:
            pair_dic = inp_indx.readlines()
            word_index = {}
            for pair in pair_dic:
                word, index = pair.replace('\n', '').split('\t')
                word_index[word] = index
            print(index)
            index_word = {v: k for k, v in word_index.items()}
            # print(index_word['18590'])
            del pair_dic

            emb_index = inp_dic.readlines()
            emb_vec = inp_vec.readlines()
            word_vectors = {}
            for index, vec in zip(emb_index, emb_vec):
                # print(index.replace('\n', ''))
                word = index_word[str(index.replace('\n', ''))]
                word_vectors[word] = np.array(list((map(float, vec.split()))))

            del emb_index
            del emb_vec
        all_txt = []
        all_lable = []
        with open(data_path + '20ng', 'r', encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            for id in all_lines:
                data_dic = json.loads(id)
                all_txt.append(data_dic["text"])
                all_lable.append(data_dic["cluster"])
            all_lines = all_txt
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())
            unigram = {}
            for item in word_count.items():
                unigram[item[0]] = item[1] / total_count

            all_vector_representation = np.zeros(shape=(18845, 48))
            for i, line in enumerate(all_lines):
                word_sentence = nltk.word_tokenize(line)

                sent_rep = np.zeros(shape=[48, ])
                j = 0
                for word in word_sentence:
                    try:
                        wv = word_vectors[word]
                        j = j + 1
                    except KeyError:
                        continue

                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                if j != 0:
                    all_vector_representation[i] = sent_rep / j
                else:
                    all_vector_representation[i] = sent_rep

        pca = PCA(n_components=1)
        pca.fit(all_vector_representation)
        pca = pca.components_

        XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

        XX = XX1

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)

        # with open(data_path + 'label_StackOverflow.txt',encoding='utf-8') as label_file:
        #     y = np.array(list((map(int, label_file.readlines()))))
        #     print(y.dtype)
        print('data have loaded')
        return XX, all_lable

def load_stackoverflow(data_path='data/stackoverflow/',type='word2v'):
    if type=='bert':

        # load SO embedding
        with open(data_path + 'vocab_withIdx.dic', 'r', encoding='utf-8') as inp_indx, \
                open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
                open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
            pair_dic = inp_indx.readlines()
            word_index = {}
            for pair in pair_dic:
                word, index = pair.replace('\n', '').split('\t')
                word_index[word] = index

            index_word = {v: k for k, v in word_index.items()}

            del pair_dic

            emb_index = inp_dic.readlines()
            emb_vec = inp_vec.readlines()
            word_vectors = {}
            for index, vec in zip(emb_index, emb_vec):
                word = index_word[index.replace('\n', '')]
                word_vectors[word] = np.array(list((map(float, vec.split()))))

            del emb_index
            del emb_vec

        with open(data_path + 'title_StackOverflow.txt', 'r',encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())
            unigram = {}
            for item in word_count.items():
                unigram[item[0]] = item[1] / total_count

            all_vector_representation = np.zeros(shape=(20000, 48))
            for i, line in enumerate(all_lines):
                word_sentence = nltk.word_tokenize(line)

                sent_rep = np.zeros(shape=[48, ])
                j = 0
                for word in word_sentence:
                    try:
                        wv = word_vectors[word]
                        j = j + 1
                    except KeyError:
                        continue

                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                if j != 0:
                    all_vector_representation[i] = sent_rep / j
                else:
                    all_vector_representation[i] = sent_rep

        pca = PCA(n_components=1)
        pca.fit(all_vector_representation)
        pca = pca.components_

        XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

        XX = XX1

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)

        with open(data_path + 'label_StackOverflow.txt',encoding='utf-8') as label_file:
            y = np.array(list((map(int, label_file.readlines()))))
            print(y.dtype)
        print('data have loaded')
        return XX, y
    if type=='word2v':
        # load SO embedding
        with open(data_path + 'vocab_withIdx.dic', 'r', encoding='utf-8') as inp_indx, \
                open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
                open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
            pair_dic = inp_indx.readlines()
            word_index = {}
            for pair in pair_dic:
                word, index = pair.replace('\n', '').split('\t')
                word_index[word] = index

            index_word = {v: k for k, v in word_index.items()}

            del pair_dic

            emb_index = inp_dic.readlines()
            emb_vec = inp_vec.readlines()
            word_vectors = {}
            for index, vec in zip(emb_index, emb_vec):
                word = index_word[index.replace('\n', '')]
                word_vectors[word] = np.array(list((map(float, vec.split()))))

            del emb_index
            del emb_vec

        with open(data_path + 'title_StackOverflow.txt', 'r', encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())
            unigram = {}
            for item in word_count.items():
                unigram[item[0]] = item[1] / total_count

            all_vector_representation = np.zeros(shape=(20000, 48))
            for i, line in enumerate(all_lines):
                word_sentence = nltk.word_tokenize(line)

                sent_rep = np.zeros(shape=[48, ])
                j = 0
                for word in word_sentence:
                    try:
                        wv = word_vectors[word]
                        j = j + 1
                    except KeyError:
                        continue

                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                if j != 0:
                    all_vector_representation[i] = sent_rep / j
                else:
                    all_vector_representation[i] = sent_rep

        pca = PCA(n_components=1)
        pca.fit(all_vector_representation)
        pca = pca.components_

        XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

        XX = XX1

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)

        with open(data_path + 'label_StackOverflow.txt', encoding='utf-8') as label_file:
            y = np.array(list((map(int, label_file.readlines()))))
            print(y.dtype)
        print('data have loaded')
        return XX, y


def load_search_snippet2(data_path='data/SearchSnippets/new/',type='word2v'):
    if type=='bert':

        mat = scipy.io.loadmat(data_path + 'SearchSnippets-STC2.mat')

        emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
        emb_vec = mat['vocab_emb_Word2vec_48']
        y = np.squeeze(mat['labels_All'])

        del mat

        rand_seed = 0

        # load SO embedding
        with open(data_path + 'SearchSnippets_vocab2idx.dic', 'r',encoding='utf-8') as inp_indx:
            pair_dic = inp_indx.readlines()
            word_index = {}
            for pair in pair_dic:
                word, index = pair.replace('\n', '').split('\t')
                word_index[word] = index

            index_word = {v: k for k, v in word_index.items()}

            del pair_dic

            word_vectors = {}
            for index, vec in zip(emb_index, emb_vec.T):
                word = index_word[str(index)]
                word_vectors[word] = vec

            del emb_index
            del emb_vec

        with open(data_path + 'SearchSnippets.txt', 'r',encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            all_lines = [line for line in all_lines]
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())
            unigram = {}
            for item in word_count.items():
                unigram[item[0]] = item[1] / total_count

            all_vector_representation = np.zeros(shape=(12340, 48))
            for i, line in enumerate(all_lines):
                word_sentence = nltk.word_tokenize(line)

                sent_rep = np.zeros(shape=[48, ])
                j = 0
                for word in word_sentence:
                    try:
                        wv = word_vectors[word]
                        j = j + 1
                    except KeyError:
                        continue

                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                if j != 0:
                    all_vector_representation[i] = sent_rep / j
                else:
                    all_vector_representation[i] = sent_rep

        svd = TruncatedSVD(n_components=1, n_iter=20)
        svd.fit(all_vector_representation)
        svd = svd.components_

        XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)
        print('data have loaded')
        return XX, y
    if type=='word2v':
        mat = scipy.io.loadmat(data_path + 'SearchSnippets-STC2.mat')

        emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
        emb_vec = mat['vocab_emb_Word2vec_48']
        y = np.squeeze(mat['labels_All'])

        del mat

        rand_seed = 0

        # load SO embedding
        with open(data_path + 'SearchSnippets_vocab2idx.dic', 'r', encoding='utf-8') as inp_indx:
            pair_dic = inp_indx.readlines()
            word_index = {}
            for pair in pair_dic:
                word, index = pair.replace('\n', '').split('\t')
                word_index[word] = index

            index_word = {v: k for k, v in word_index.items()}

            del pair_dic

            word_vectors = {}
            for index, vec in zip(emb_index, emb_vec.T):
                word = index_word[str(index)]
                word_vectors[word] = vec

            del emb_index
            del emb_vec

        with open(data_path + 'SearchSnippets.txt', 'r', encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            all_lines = [line for line in all_lines]
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())
            unigram = {}
            for item in word_count.items():
                unigram[item[0]] = item[1] / total_count

            all_vector_representation = np.zeros(shape=(12340, 48))
            for i, line in enumerate(all_lines):
                word_sentence = nltk.word_tokenize(line)

                sent_rep = np.zeros(shape=[48, ])
                j = 0
                for word in word_sentence:
                    try:
                        wv = word_vectors[word]
                        j = j + 1
                    except KeyError:
                        continue

                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                if j != 0:
                    all_vector_representation[i] = sent_rep / j
                else:
                    all_vector_representation[i] = sent_rep

        svd = TruncatedSVD(n_components=1, n_iter=20)
        svd.fit(all_vector_representation)
        svd = svd.components_

        XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)
        print('data have loaded')
        return XX, y



def load_biomedical(data_path='data/Biomedical/',type='word2v'):
    if type=='bert':
        mat = scipy.io.loadmat(data_path + 'Biomedical-STC2.mat')

        emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
        emb_vec = mat['vocab_emb_Word2vec_48']
        y = np.squeeze(mat['labels_All'])

        del mat

        rand_seed = 0

        # load SO embedding
        with open(data_path + 'Biomedical_vocab2idx.dic', 'r',encoding='utf-8') as inp_indx:
            # open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
            # open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
            pair_dic = inp_indx.readlines()
            word_index = {}
            for pair in pair_dic:
                word, index = pair.replace('\n', '').split('\t')
                word_index[word] = index

            index_word = {v: k for k, v in word_index.items()}

            del pair_dic

            word_vectors = {}
            for index, vec in zip(emb_index, emb_vec.T):
                word = index_word[str(index)]
                word_vectors[word] = vec

            del emb_index
            del emb_vec

        with open(data_path + 'Biomedical.txt', 'r',encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            # print(sum([len(line.split()) for line in all_lines])/20000) #avg length
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())
            unigram = {}
            for item in word_count.items():
                unigram[item[0]] = item[1] / total_count

            all_vector_representation = np.zeros(shape=(20000, 48))
            for i, line in enumerate(all_lines):
                word_sentence = nltk.word_tokenize(line)

                sent_rep = np.zeros(shape=[48, ])
                j = 0
                for word in word_sentence:
                    try:
                        wv = word_vectors[word]
                        j = j + 1
                    except KeyError:
                        continue

                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                if j != 0:
                    all_vector_representation[i] = sent_rep / j
                else:
                    all_vector_representation[i] = sent_rep

        svd = TruncatedSVD(n_components=1, random_state=rand_seed, n_iter=20)
        svd.fit(all_vector_representation)
        svd = svd.components_
        XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)
        print('data have loaded')
        return XX, y
    if type=='word2v':
        mat = scipy.io.loadmat(data_path + 'Biomedical-STC2.mat')

        emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
        emb_vec = mat['vocab_emb_Word2vec_48']
        y = np.squeeze(mat['labels_All'])

        del mat

        rand_seed = 0

        # load SO embedding
        with open(data_path + 'Biomedical_vocab2idx.dic', 'r', encoding='utf-8') as inp_indx:
            # open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
            # open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
            pair_dic = inp_indx.readlines()
            word_index = {}
            for pair in pair_dic:
                word, index = pair.replace('\n', '').split('\t')
                word_index[word] = index

            index_word = {v: k for k, v in word_index.items()}

            del pair_dic

            word_vectors = {}
            for index, vec in zip(emb_index, emb_vec.T):
                word = index_word[str(index)]
                word_vectors[word] = vec

            del emb_index
            del emb_vec

        with open(data_path + 'Biomedical.txt', 'r', encoding='utf-8') as inp_txt:
            all_lines = inp_txt.readlines()[:-1]
            # print(sum([len(line.split()) for line in all_lines])/20000) #avg length
            text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
            word_count = Counter(text_file.split())
            total_count = sum(word_count.values())
            unigram = {}
            for item in word_count.items():
                unigram[item[0]] = item[1] / total_count

            all_vector_representation = np.zeros(shape=(20000, 48))
            for i, line in enumerate(all_lines):
                word_sentence = nltk.word_tokenize(line)

                sent_rep = np.zeros(shape=[48, ])
                j = 0
                for word in word_sentence:
                    try:
                        wv = word_vectors[word]
                        j = j + 1
                    except KeyError:
                        continue

                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                if j != 0:
                    all_vector_representation[i] = sent_rep / j
                else:
                    all_vector_representation[i] = sent_rep

        svd = TruncatedSVD(n_components=1, random_state=rand_seed, n_iter=20)
        svd.fit(all_vector_representation)
        svd = svd.components_
        XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)
        print('data have loaded')
        return XX, y


def load_data(dataset_name,type='word2v'):
    print('load dataset: '+dataset_name+", embedding type: "+type)
    if dataset_name == 'stackoverflow':
        return load_stackoverflow(type=type)
    elif dataset_name == 'biomedical':
        return load_biomedical(type=type)
    elif dataset_name == 'search_snippets':
        return load_search_snippet2(type=type)
    elif dataset_name == '20ng':
        return load_20ng(type=type)
    elif dataset_name == 'Tweet':
        return load_tweet(type=type)
    elif dataset_name == 'News':
        return load_News(type=type)
    else:
        raise Exception('dataset not found...')
