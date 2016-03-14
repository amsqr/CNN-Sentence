

import sys
import csv
import re
import numpy as np
import cPickle
import time
from sklearn.decomposition import PCA

csv.field_size_limit(sys.maxint)


####//////////#####################
# functions to get training, validation and test data

def get_sen_dic(filename):
    sen_dic = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n')
            sep = re.split('\t', line)
            sen_dic[sep[1]] = int(sep[0])
    return sen_dic

def get_sen_split(filename):
    sen_split = []
    sen_split.append(0)
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n')
            sep = re.split(',', line)
            sen_split.append(int(sep[1]))
    return sen_split

def get_phrase_label(filename):
    phrase_label = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n')
            sep = line.split('|')
            # note: re.split does not work!!! sep = re.split('|', line) failed.
            label = float(sep[1])
            if (label/0.2) % 1 == 0 and label != 0:
                phrase_label.append(int(label/0.2) - 1)
            else:
                phrase_label.append(int(label/0.2))
    return phrase_label

def get_data(filename, sen_dic, sen_split, phrase_label):
    vocab = {}
    traindata = []
    valdata = []
    testdata = []
    word_cnt = 1
    max_sen_len = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n')
            sep = line.split('|')
            phrase = re.split('\s+', sep[0])
            phrase_vec = []
            for word in phrase:
                if word not in vocab:
                    vocab[word] = word_cnt
                    phrase_vec.append(word_cnt)
                    word_cnt += 1
                else:
                    phrase_vec.append(vocab[word])
            max_sen_len = max(max_sen_len, len(phrase_vec))
            phrase_vec.append(phrase_label[int(sep[1])])
            if sep[0] in sen_dic:
                if sen_split[sen_dic[sep[0]]] == 1:
                    traindata.append(phrase_vec)
                elif sen_split[sen_dic[sep[0]]] == 2:
                    valdata.append(phrase_vec)
                else:
                    testdata.append(phrase_vec)
            else:
                traindata.append(phrase_vec)
    return (vocab, [traindata, valdata, testdata], max_sen_len) 

def load_bin_vec(fname, vocab):
    """
    Loads 300 dimensional word vecs from Google (Mikolov) word2vec
    """
    word_vec_dic = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vec_dic[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vec_dic


def get_word_vec(vocab, word_vec_dic):
    vocab_size = len(vocab)
    word_dim = len(word_vec_dic.itervalues().next())
    word_vec = np.zeros(shape=(vocab_size + 1, word_dim)) 
    word_vec[0] = np.zeros(word_dim)
    word_in_google = 0 
    for word in vocab:
        if word in word_vec_dic:
            word_vec[vocab[word]] = word_vec_dic[word]
            word_in_google += 1
        else:
            word_vec[vocab[word]] = np.random.uniform(-0.25, 0.25, word_dim)
    print 'there are ' + str(word_in_google) + '/' + str(vocab_size) +  ' words found in google word2vec'
    return word_vec

def interact_matvec(all_data, word_vec, added_dim=10):
    pca = PCA(n_components=added_dim, copy=False)
    pca_all_data = []
    word_dim = len(word_vec[0])
    pca_cnt = 0
    pca_incre = 2000
    new_time = 0
    olde_time = 0
    for data in all_data:
        pca_data = []
        for sen in data:
            if len(sen) > 2:
                pca_sen = sen[:-1]
                interaction = np.zeros(shape=(word_dim, word_dim))
                for (i,j) in zip(sen[:-1], sen[:-1][1:]):
                    # pairwise column row vector multiplication into matrix and summation.
                    interaction += np.outer(word_vec[i], word_vec[j]).reshape(word_dim, word_dim)
                # reshape is needed here. i thought pca extracts fixed feature numbers, but it's not the case
                interaction = np.dot(interaction, 1.0/(len(sen)-2))
                added_words = pca.fit_transform(interaction)
                pca_cnt += 1
                if pca_cnt % pca_incre == 0:
                    print 'I have done ' + str(pca_cnt) + ' pca steps!'
                    old_time = new_time
                    new_time = time.time()
                    print 'took ' + str(new_time-old_time) + ' s to run ' + str(pca_incre) + ' steps'
                added_words = added_words.reshape(added_dim, word_dim)
                for new_word in added_words:
                    np.concatenate((word_vec, [new_word]), axis=0)
                    pca_sen.append(len(word_vec)-1)
                pca_sen.append(sen[-1])
            else:
                pca_sen = sen
            pca_data.append(pca_sen)
        pca_all_data.append(pca_data)
    return (pca_all_data, word_vec)

def pad_sentence(senset, max_sen_len, filter_heights):
    pad = max(filter_heights) - 1
    padded_senset = []
    for vec_full in senset:
        vec = vec_full[:-1]
        x = []
        for i in range(pad):
            x.append(0)
        x.extend(vec)
        while len(x) < max_sen_len + 2*pad:
            x.append(0)
        x.append(vec_full[-1])
        padded_senset.append(x)
    return padded_senset


if __name__=="__main__":

    print 'start data processing...'
    
    dirc = '//Users//NINI//Dropbox (MIT)//NLP//DataSets//STS//'
    input_f = 'datasetSentences.txt'
    sen_dic = get_sen_dic(dirc + input_f)
    
    input_f = 'datasetSplit.txt'
    sen_split = get_sen_split(dirc + input_f)
    
    input_f = 'sentiment_labels.txt'
    phrase_label = get_phrase_label(dirc + input_f)
    
    input_f = 'dictionary.txt'
    vocab, data, max_sen_len = get_data(dirc + input_f, sen_dic, sen_split, phrase_label)
    print 'vocab and raw data are processed!'
    print 'max length of sentence is ' + str(max_sen_len) + '.'
    
    print 'creating word vectors...'
    input_f = 'Z:\MIT courses\6.867\6.867 project\word2vec\GoogleNews-vectors-negative300.bin'
    word_vec_dic = load_bin_vec(input_f, vocab)
    
    word_vec = get_word_vec(vocab, word_vec_dic)
    print 'word vector created!'

    output_f = "CWCNN_baseline.p"
    cPickle.dump([data, word_vec, max_sen_len], open(output_f, 'wb'))

