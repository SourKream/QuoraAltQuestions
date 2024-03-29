import argparse
import codecs
import logging
import numpy as np
import os
import pdb
import pickle
import sys
import theano

np.random.seed(1337)
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.callbacks import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.merge import *
from keras.layers import *
from sklearn.metrics import *
from Utils import *
from datetime import datetime
# from matplotlib import cm

def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=75, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=10, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=128, dest="batch_size", type=int)
    parser.add_argument('-xmaxlen', action="store", default=12, dest="xmaxlen", type=int)
    parser.add_argument('-ymaxlen', action="store", default=12, dest="ymaxlen", type=int)
    parser.add_argument('-nopad', action="store", default=False, dest="no_padding", type=bool)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-decay', action="store", default=0.2, dest='decay', type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-verbose', action="store", default=False, dest="verbose", type=bool)
    parser.add_argument('-l2', action="store", default=0.01, dest="l2", type=float)
    parser.add_argument('-dropout', action="store", default=0.1, dest="dropout", type=float)
    parser.add_argument('-local', action="store", default=False, dest="local", type=bool)
    parser.add_argument('-embd', action="store", default=200, dest='embd_size', type=int)
    opts = parser.parse_args(sys.argv[1:])
    print "lstm_units", opts.lstm_units
    print "epochs", opts.epochs
    print "batch_size", opts.batch_size
    print "xmaxlen", opts.xmaxlen
    print "ymaxlen", opts.ymaxlen
    print "no_padding", opts.no_padding
    print "regularization factor", opts.l2
    print "dropout", opts.dropout
    print "LR", opts.lr
    print "Decay", opts.decay
    print "Embedding Size", opts.embd_size
    return opts

def get_H_i(i):  
    # get element i from time dimension
    def get_X_i(X):
        return X[:,i,:];
    return get_X_i

def get_H_n(X):
    # get last element from time dimension
    return X[:, -1, :]

def get_H_premise(X):
    # get elements 1 to L from time dimension
    xmaxlen = K.params['xmaxlen']
    return X[:, :xmaxlen, :] 

def get_H_hypothesis(X):
    # get elements L+1 to N from time dimension
    xmaxlen = K.params['xmaxlen']
    return X[:, xmaxlen:, :]  

def weighted_average_pooling(X):
    # Matrix A (BatchSize, Time, Feature) 
    # Matrix Alpha (BatchSize, Time, 1)
    # Matrix A Averaged along Time axis according to weights Alpha
    #    
    # Input X : (BatchSize, Time, Feature + 1) Formed by concatinating A and Alpha along Feature axis
    # Output Y : (BatchSize, Feature) Weighted average of A along time according to Alpha

    A = X[:,:,:-1]
    Alpha = X[:,:,-1]
    A = K.permute_dimensions(A, (0,2,1))  
    # Now A is (None,k,L) and Alpha is always (None,L,1)
    return K.T.batched_dot(A, Alpha)

def build_model(opts, verbose=False):

    # LSTM Output Dimension
    k = 2 * opts.lstm_units

    # Premise Length
    L = opts.xmaxlen

    # Premise + Delimiter + Hypothesis Length
    N = opts.xmaxlen + opts.ymaxlen + 1

    print "Premise Length : ", L
    print "Total Length   : ", N

    input_layer = Input(shape=(N,), dtype='int32', name="Input Layer")

    emb_layer = Embedding(opts.vocab_size, 
                            opts.embd_size,
                            input_length = N,
                            name = "Embedding Layer") (input_layer)    
    emb_layer = SpatialDropout1D(opts.dropout)(emb_layer)

    LSTMEncoding = Bidirectional(LSTM(opts.lstm_units,
                                    return_sequences = True, 
                                    name="LSTM Layer")) (emb_layer)

    LSTMEncoding = Dropout(opts.dropout, name="Dropout LSTM Layer")(LSTMEncoding)

    h_n = Lambda(get_H_n, output_shape=(k,), name = "h_n")(LSTMEncoding)

    Y = Lambda(get_H_premise, output_shape = (L, k), name = "Y")(LSTMEncoding)
    Y = TimeDistributed(Dense(k, kernel_regularizer = l2(0.01)))(Y)

    h_hypo = Lambda(get_H_hypothesis, output_shape = (N-L, k), name = "h_hypo")(LSTMEncoding)
    h_hypo = TimeDistributed(Dense(k, kernel_regularizer = l2(0.01)))(h_hypo)

    ## Init Dense Weights
    alpha_init_weight = ((2.0/np.sqrt(k)) * np.random.rand(k,1)) - (1.0 / np.sqrt(k))
    alpha_init_bias = ((2.0) * np.random.rand(1,)) - (1.0)
    Tan_Wr_init_weight = 2*(1/np.sqrt(k))*np.random.rand(k,k) - (1/np.sqrt(k))
    Tan_Wr_init_bias = 2*(1/np.sqrt(k))*np.random.rand(k,) - (1/np.sqrt(k))
    Wr_init_weight = 2*(1/np.sqrt(k))*np.random.rand(k,k) - (1/np.sqrt(k))
    Wr_init_bias = 2*(1/np.sqrt(k))*np.random.rand(k,) - (1/np.sqrt(k))

    # GET R1, R2, R3, .. R_N
    for i in range(1, N-L+1):
        Wh_i = Lambda(get_H_i(i-1), output_shape=(k,))(h_hypo)

        if i == 1:
            M = Activation('tanh')(add([RepeatVector(L)(Wh_i), Y]))
        else:
            M = Activation('tanh')(add([RepeatVector(L)(Wh_i), Y, RepeatVector(L)(Wr)]))

        alpha = Reshape((L, 1), input_shape=(L,))(Activation("softmax")(Flatten()(TimeDistributed(Dense(1, weights=[alpha_init_weight, alpha_init_bias]), name='alpha'+str(i))(M))))

        r = Lambda(weighted_average_pooling, output_shape=(k,), name="r"+str(i))(concatenate([Y, alpha], axis = 2))

        if i != 1:
            r = add([r, Tan_Wr])

        if i != (N-L):

            Tan_Wr = Dense(k, kernel_regularizer = l2(0.01),
                    activation = 'tanh',
                    name='Tan_Wr'+str(i), 
                    weights = [Tan_Wr_init_weight, Tan_Wr_init_bias])(r)
            Wr = Dense(k, kernel_regularizer = l2(0.01), 
                            name = 'Wr'+str(i), 
                            weights = [Wr_init_weight, Wr_init_bias])(r)


    r = Dense(k, kernel_regularizer = l2(0.01))(r) 
    h_n = Dense(k, kernel_regularizer = l2(0.01))(h_n)

    h_star = Activation('tanh')(add([r, h_n]))

    output_layer = Dense(1, activation='sigmoid', name="Output Layer")(h_star)

    model = Model(inputs = input_layer, outputs = output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(opts.lr))
    print "Model Compiled"

    return model

def compute_acc(X, Y, model):
    scores = model.predict(X)
    plabels = np.round(scores)
    tlabels = np.matrix(Y).transpose()
    p, r, f, _ = precision_recall_fscore_support(tlabels, plabels)

    return p[1], r[1], f[1]

def getConfig(opts):
    conf = [opts.lstm_units, opts.embd_size, opts.vocab_size, opts.lr, opts.l2, opts.xmaxlen]
    return "_".join(map(lambda x: str(x), conf))

class WeightSharing(Callback):
    def __init__(self, shared):
        self.shared = shared

    def find_layer_by_name(self, name):
        for l in self.model.layers:
            if l.name == name:
                return l

    def on_batch_end(self, batch, logs={}):
        weights = np.mean([self.find_layer_by_name(n).get_weights()[0] for n in self.shared],axis=0)
        biases = np.mean([self.find_layer_by_name(n).get_weights()[1] for n in self.shared],axis=0)
        for n in self.shared:
            self.find_layer_by_name(n).set_weights([weights, biases])

class Metrics(Callback):
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x 
        self.test_y = test_y

    def on_epoch_end(self, epochs, logs={}):
        train_pre, train_rec, train_f = compute_acc(self.train_x, self.train_y, self.model)
        test_pre, test_rec, test_f  = compute_acc(self.test_x, self.test_y, self.model)
        print "\n\nTraining -> Precision: ", train_pre, "\t Recall: ", train_rec, "\t F-Score: ", train_f
        print "Validation  -> Precision: ", test_pre,  "\t Recall: ", test_rec,  "\t F-Score: ", test_f, "\n"


class WeightSave(Callback):
    def __init__(self, path, config_str):
        self.path = path
        self.config_str = config_str

    def on_epoch_end(self,epochs, logs={}):
        self.model.save_weights(self.path + self.config_str +"_"+ str(epochs) +  ".weights") 

if __name__ == "__main__":

    options = get_params()

    # ## PARAMS
    # options.lstm_units = 15
    # options.batch_size = 128
    # options.embd = 20

    dataPath = './Data/'

    train = [line.strip().split('\t') for line in open(dataPath + 'Train.txt')]
    val = [line.strip().split('\t') for line in open(dataPath + 'Val.txt')]
    # test = [line.strip().split('\t') for line in open(dataPath + 'Test.txt')]
    vocab = get_vocab(train)

    options.vocab_size = len(vocab)
    print "Vocab Size : ", len(vocab)

    X_train, Y_train, labels_train = load_data(train, vocab)
    X_val, Y_val, labels_val = load_data(val, vocab)
    # X_test,  Y_test,  labels_test  = load_test_data(test,  vocab)
   
    params = {'xmaxlen': options.xmaxlen}
    setattr(K, 'params', params)
   
    XMAXLEN = options.xmaxlen
    YMAXLEN = options.ymaxlen
    X_train = pad_sequences(X_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'pre')
    X_val = pad_sequences(X_val, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'pre')
    # X_test  = pad_sequences(X_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'pre')
    Y_train = pad_sequences(Y_train, maxlen = YMAXLEN, value = vocab["pad_tok"], padding = 'post')
    Y_val = pad_sequences(Y_val, maxlen = YMAXLEN, value = vocab["pad_tok"], padding = 'post')
    # Y_test  = pad_sequences(Y_test,  maxlen = YMAXLEN, value = vocab["pad_tok"], padding = 'post')
   
    net_train = concat_in_out(X_train, Y_train, vocab)
    net_val = concat_in_out(X_val, Y_val, vocab)
    # net_test  = concat_in_out(X_test , Y_test , vocab)

    assert net_train[0][options.xmaxlen] == vocab['delimiter']

    # options.load_save = True
    # MODEL_WGHT = './Models/IPAModel_75_200_539_0.001_0.005_12_adam_4.weights'
    # MODEL_WGHT = './Models/IPAModel_15_20_539_0.001_0.005_12_adam_9.weights'

    if options.load_save and os.path.exists(MODEL_WGHT):
        print("Loading pre-trained model from ", MODEL_WGHT)
        model = build_model(options)
        model.load_weights(MODEL_WGHT)

    else:
        print 'Building model'
        model = build_model(options)

        print 'Training New Model'
        group1 = ['Tan_Wr'+str(i) for i in range(1, options.ymaxlen+1)]
        group2 = ['Wr'+str(i) for i in range(1, options.ymaxlen+1)]
        group3 = ['alpha'+str(i) for i in range(1, options.ymaxlen+2)]

        ModelSaveDir = "./Models/QuoBaseModel_"
        save_weights = WeightSave(ModelSaveDir, getConfig(options))
        metrics_callback = Metrics(net_train, labels_train, net_val, labels_val)

        history = model.fit(x = net_train, 
                            y = labels_train,
                            batch_size = options.batch_size,
                            epochs = options.epochs,
                            callbacks = [WeightSharing(group1), WeightSharing(group2), WeightSharing(group3), save_weights, metrics_callback])
