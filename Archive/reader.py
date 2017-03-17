# coding: utf-8
import json
from myutils import *
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

def load_data(train,vocab):
    X,Y,Z = [],[],[]
    for p,h,l in train:
        p = map_to_idx(tokenize(p),vocab)
        h = map_to_idx(tokenize(h),vocab)
        X += [p]
        Y += [h]
        Z += [l]
    return X,Y,Z

def get_vocab(data):
    vocab = Counter()
    for ex in data:
        tokens = tokenize(ex[0])
        tokens += tokenize(ex[1])
        vocab.update(tokens)
    lst = ["unk", "delimiter", "pad_tok"] + [ x for x, y in vocab.iteritems() if y > 0]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    return vocab

if __name__=="__main__":

    train = [l.strip().split('\t') for l in open('Data/train.txt')]
    dev = [l.strip().split('\t') for l in open('Data/dev.txt')]
    test = [l.strip().split('\t') for l in open('Data/test.txt')]
    labels = {0:'not_duplicate',1:'duplicate'}

    vocab = get_vocab(train)
    # X_train,Y_train,Z_train = load_data(train,vocab)
    X_dev,Y_dev,Z_dev = load_data(dev,vocab)
    # print len(X_train), X_train[0]
    # print len(X_dev), X_dev[0]
