import pickle
import os

class helper_functions():

    def save_dict(dict, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open (os.path.join(path +'.pkl'), "wb") as f:
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