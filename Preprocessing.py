import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from transformers import RobertaTokenizer
import numpy as np
from tqdm import tqdm

from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.linear_model import LogisticRegression

from gensim import downloader
from numpy.random import default_rng
from scipy.linalg import null_space

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pickle
import os

DATA_PATH = ".\data"
PLOTS_PATH = ".\plots"
SEED = 5
np.random.seed(SEED)


class Preprocessing:

    def __init__(self, path, rep_model='', attributes=[]):
        print('Preprocessing is started')
        self.data = pd.read_json(path, lines=True)
        self.attributes = attributes
        if rep_model == '':
            print('download glove-twitter-200')
            rep_model = downloader.load('glove-twitter-200')
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        if not os.path.exists(PLOTS_PATH):
            os.makedirs(PLOTS_PATH)
        self.rep_model = rep_model
        # pre-trained model which tags words to their part-of-speech.
        self.tagger = SequenceTagger.load("flair/pos-english-fast")
        self.df_words_poses = self.get_df_words_poses()
        self.word_2_vec = self.get_word_2_vec()
        self.word_2_pos = {}
        self.projected_word_2_vec = {}
        for attribute in self.attributes:
            self.word_2_pos[attribute] = self.get_word_2_pos(attribute)
            self.projected_word_2_vec[attribute] = self.get_projected_word_2_vec(
                attribute)
        self.create_train_test_set()
        self.count_plot()
        self.length_distribution_plot()
        self.plot_top_k_poses()

    def count_plot(self):
        sns.countplot(data=self.data, x='is_sarcastic')
        plt.savefig(os.path.join(PLOTS_PATH, f'count_plot'))
        plt.clf()

    def length_distribution_plot(self):
        def length(phrase):
            return len(phrase.split())

        self.data["length"] = self.data["headline"].apply(length)
        plt.figure()
        sns.displot(data=self.data, x="length", kde=True)
        plt.title("distribution of number of words in headlines")
        plt.savefig(os.path.join(PLOTS_PATH, f'length_distribution'))
        plt.clf()

    def get_words_poses_lists(self):
        headlines = self.data['headline']
        poses = []
        words = []
        for _, sentence in tqdm(enumerate(headlines)):
            sentence = Sentence(sentence)
            self.tagger.predict(sentence)
            pos_dic = sentence.to_dict(tag_type='pos')
            for word in pos_dic['entities']:
                if not word['text'] in words:
                    words.append(word['text'])
                    poses.append(word['labels'][0].value)
        return words, poses

    def create_df_words_poses(self, file_name='data_pos.csv'):
        words, poses = self.get_words_poses_lists()
        df = pd.DataFrame(list(zip(words, poses)), columns=['words', 'pos'])
        df.to_csv(file_name)
        return df

    def plot_top_k_poses(self, k=10):
        top_k_pos = self.df_words_poses['pos'].value_counts()[:k]
        top_k_pos = pd.DataFrame({"pos": top_k_pos.index, "count": top_k_pos})
        sns.barplot(x="pos", y='count', data=top_k_pos)
        plt.title(f'Top {k} parts-of-speech')
        plt.savefig(os.path.join(PLOTS_PATH, f'top_{k}_parts_of_speech'))
        plt.clf()

    def create_word_2_vec(self):
        print('create word 2 vec dictionary...')
        words = self.df_words_poses['words']
        word_2_vec = {}
        for _, word in tqdm(enumerate(words)):
            if not word in word_2_vec.keys() and self.rep_model.has_index_for(word.lower()):
                word_vec = self.rep_model.get_vector(word.lower())
                word_2_vec[word.lower()] = word_vec
        with open(os.path.join(DATA_PATH, 'word_2_vec_glove.pkl'), "wb") as f:
            pickle.dump(word_2_vec, f)

    def create_word_2_pos(self, attribute):
        print('create word 2 pos dictionary...')
        word_2_pos = {}
        words = self.df_words_poses['words']
        poses = self.df_words_poses['pos']
        for _, word in tqdm(enumerate(self.word_2_vec.keys())):
            pos_of_word = poses[words[words == word].index.values[0]]
            word_2_pos[word.lower()] = 1 if pos_of_word == attribute else 0
        with open(os.path.join(DATA_PATH, 'word_2_label_' + attribute + '.pkl'), "wb") as f:
            pickle.dump(word_2_pos, f)

    def get_df_words_poses(self, file_name='data_pos.csv'):
        if not os.path.exists(file_name):
            print('create data-pos data frame..')
            self.create_df_words_poses(file_name)
        df_words_poses = pd.read_csv(file_name)
        return df_words_poses

    def get_word_2_vec(self, file_name='word_2_vec_glove.pkl'):
        if not os.path.exists(os.path.join(DATA_PATH, file_name)):
            print('create word 2 vec dictionary...')
            self.create_word_2_vec()
        with open(os.path.join(DATA_PATH, file_name), "rb") as f:
            word_2_vec = pickle.load(f)
        return word_2_vec

    def get_word_2_pos(self, attribute):
        file_name = f'word_2_label_{attribute}.pkl'
        if not os.path.exists(os.path.join(DATA_PATH, file_name)):
            self.create_word_2_pos(attribute)
        with open(os.path.join(DATA_PATH, file_name), "rb") as f:
            word_2_vec = pickle.load(f)
        return word_2_vec

    def INLP(self, attribute, iter=10):
        vecs = np.array(list(self.word_2_vec.values()))
        labels = list(self.word_2_pos[attribute].values())
        print(
            f'We aim to reach: ~{1 - sum(labels) / len(labels):.3f} accuracy ')
        accuracy_list = []
        X_projected = vecs.T
        for i in tqdm(np.arange(iter)):
            clf = LogisticRegression(random_state=5).fit(X_projected.T, labels)
            w = clf.coef_
            predictions = clf.predict(X_projected.T)
            accuracy = accuracy_score(predictions, labels)
            print(f'Accuracy iter {i+1}: {accuracy:.3f}')
            b = null_space(w)
            p_null_space = b @ b.T
            X_projected = p_null_space @ X_projected
            accuracy_list.append(accuracy)
        plt.plot(np.arange(1, iter + 1), accuracy_list)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'INLP_{attribute}')
        plt.grid()
        plt.savefig(os.path.join(PLOTS_PATH, f'INLP_{attribute}'))
        plt.clf()
        return X_projected

    def create_projected_word_2_vec(self, attribute):
        print('create projected word 2 vec dictionary...')
        X_projected = self.INLP(attribute)
        projected_word_2_vec = {}
        for idx, word in enumerate(self.word_2_vec):
            projected_word_2_vec[word] = X_projected[:, idx]
        with open(os.path.join(DATA_PATH, 'proj_word_2_vec_' + attribute + '.pkl'), "wb") as f:
            pickle.dump(projected_word_2_vec, f)

    def get_projected_word_2_vec(self, attribute):
        file_name = f'proj_word_2_vec_{attribute}.pkl'
        if not os.path.exists(os.path.join(DATA_PATH, file_name)):
            self.create_projected_word_2_vec(attribute)
        with open(os.path.join(DATA_PATH, file_name), "rb") as f:
            proj_word_2_vec = pickle.load(f)
        return proj_word_2_vec

    def create_train_test_set(self):
        train_data, test_data = train_test_split(self.data, test_size=0.2)
        train_sen_2_vec, train_sen_2_label = self.get_sentences_maps(
            train_data, self.word_2_vec)
        test_sen_2_vec, test_sen_2_label = self.get_sentences_maps(
            test_data, self.word_2_vec)

        for attribute in self.attributes:
            proj_test_sen_2_vec, proj_test_sen_2_label = self.get_sentences_maps(test_data,
                                                                                 self.projected_word_2_vec[attribute])
            self.save_dict(proj_test_sen_2_vec, os.path.join(
                DATA_PATH, f'test_sen_2_vec_{attribute}'))
            self.save_dict(proj_test_sen_2_label, os.path.join(
                DATA_PATH, f'test_sen_2_label_{attribute}'))

        self.save_dict(train_sen_2_vec, os.path.join(
            DATA_PATH, 'train_sen_2_vec'))
        self.save_dict(train_sen_2_label, os.path.join(
            DATA_PATH, 'train_sen_2_label'))
        self.save_dict(test_sen_2_vec, os.path.join(
            DATA_PATH, 'test_sen_2_vec'))
        self.save_dict(test_sen_2_label, os.path.join(
            DATA_PATH, 'test_sen_2_label'))

    def save_dict(self, dict, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}.pkl', "wb") as f:
            pickle.dump(dict, f)

    def get_sentences_maps(self, data_2_map, word_2_vec):
        sentence_id_2_vec = {}
        sentence_id_2_label = {}
        for idx, row in enumerate(data_2_map.values):
            headline = row[1].split(' ')
            sentence = []
            for word in headline:
                if word in word_2_vec.keys():
                    sentence.append(word_2_vec[word])
            if not len(sentence) == 0:
                sentence_id_2_vec[idx] = sentence
                sentence_id_2_label[idx] = row[0]
        return sentence_id_2_vec, sentence_id_2_label
