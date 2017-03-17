import sys
import re
import numpy as np
import argparse
import random
from sklearn.metrics import confusion_matrix

def tokenize(sent):
    return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]

def map_to_idx(x, vocab):
    # 0 is for UNK
    return [ vocab[w] if w in vocab else 0 for w in x  ]

def map_to_txt(x,vocab):
    textify=map_to_idx(x,inverse_map(vocab))
    return ' '.join(textify)
    
def inverse_map(vocab):
    return {v: k for k, v in vocab.items()}

def concat_in_out(X,Y,vocab):
    numex = X.shape[0] # num examples
    glue = vocab["delimiter"]*np.ones(numex).reshape(numex,1)
    inp_train = np.concatenate((X,glue,Y),axis=1)
    return inp_train

def getResults (labels, predicted):
	c = confusion_matrix(labels, predicted, [1, 0])
	a = float(c[0][0] + c[1][1])/(c[0][0]+c[1][0]+c[0][1]+c[1][1])
	p = float(c[0][0])/(c[0][0]+c[1][0])
	r = float(c[0][0])/(c[0][0]+c[0][1])
	f = 2*p*r/(p+r)
	return a*100, p*100, r*100, f
