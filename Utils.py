from sklearn.metrics import *
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re

def tokenize(sent):
    return [x.strip().lower() for x in re.split('(\W+)?', sent.strip()) if x.strip()]

def map_to_idx(x, vocab):
    return [vocab[w] if w in vocab else vocab["unk"] for w in x]

def map_to_txt(x, vocab):
    textify = map_to_idx(x, inverse_map(vocab))
    return ' '.join(textify)
    
def inverse_map(vocab):
    return {vocab[item]: item for item in vocab}

def concat_in_out(X, Y, vocab):
    numex = X.shape[0] # num examples
    glue = vocab["delimiter"]*np.ones(numex).reshape(numex,1)
    inp_train = np.concatenate((X,glue,Y),axis=1)
    return inp_train

def load_data(train, vocab, labels = {'0':0,'1':1,0:0,1:1}):
    X,Y,Z = [],[],[]
    for p,h,l in train:
        p = map_to_idx(tokenize(p), vocab)
        h = map_to_idx(tokenize(h), vocab)
        if l in labels:         
            X += [p]
            Y += [h]
            Z += [labels[l]]
    return X,Y,Z

def load_test_data(test, vocab, labels = {'0':0,'1':1,0:0,1:1}):
    X,Y = [],[]
    for p,h in test:
        p = map_to_idx(tokenize(p), vocab)
        h = map_to_idx(tokenize(h), vocab)
        X += [p]
        Y += [h]
    return X,Y

def get_vocab(data):
    vocab = Counter()
    for ex in data:
        tokens = tokenize(ex[0])
        tokens += tokenize(ex[1])
        vocab.update(tokens)
    tokens = ["unk", "delimiter", "pad_tok"] + [x for x, y in sorted(vocab.iteritems()) if y > 0]
    vocab = {y:x for x,y in enumerate(tokens)}
    return vocab

def plot_attention(matrix, title='Attention matrix', cmap=plt.cm.Blues, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=cmap)
    plt.title(title)
    fig.colorbar(cax)
    if labels:
        ax.set_xticklabels([''] + labels[1])
        ax.set_yticklabels([''] + labels[0])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

def getPRCurveKeras (x_test, y_test, model, label='DL Model'):
    p_proba = model.predict(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, p_proba)
    plt.plot(recall, precision, label='{0} (AUC = {1:0.2f})'.format(label, auc(recall, precision)))

def getAccuracy(x_test, y_test, model):
    pred = model.predict(x_test)
    pred = np.round(pred)
    s = 0
    for i in range(len(pred)):
        if pred[i][0] == y_test[i]:
            s += 1
    return s/float(len(y_test))

def getROCCurveKeras (x_test, y_test, model, label='DL Model'):
    p_proba = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, p_proba)
    plt.plot(fpr, tpr, label='{0} (AUC = {1:0.2f})'.format(label, auc(fpr, tpr)))
