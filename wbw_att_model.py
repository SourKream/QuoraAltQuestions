import numpy as np
import os
import sys
import logging
import pdb
import theano

np.random.seed(1337)
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2
from keras.callbacks import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical, accuracy
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers import *
from reader import *
from myutils import *
from datetime import datetime
from matplotlib import cm

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lstm', action="store", default=150, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-xmaxlen', action="store", default=30, dest="xmaxlen", type=int)
    parser.add_argument('-ymaxlen', action="store", default=30, dest="ymaxlen", type=int)
    parser.add_argument('-nopad', action="store", default=False, dest="no_padding", type=bool)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-decay', action="store", default=0.2, dest='decay', type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-verbose', action="store", default=False, dest="verbose", type=bool)
    parser.add_argument('-l2', action="store", default=0.0003, dest="l2", type=float)
    parser.add_argument('-dropout', action="store", default=0.1, dest="dropout", type=float)
    parser.add_argument('-local', action="store", default=True, dest="local", type=bool)
    parser.add_argument('-optimiser', action="store", default='adam', dest='optimiser', type=str)
    parser.add_argument('-embd', action="store", default=300, dest='embd', type=int)
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
    print "Optimiser", opts.optimiser
    print "Embedding Size", opts.embd
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

    # Initial Embedding (Initialise using GloVe)
#    initEmbeddings = np.load(opts.embeddings_file_path)
#
#    emb_layer = Embedding(initEmbeddings.shape[0], 
#                            initEmbeddings.shape[1],
#                            input_length = N,
#                            weights = [initEmbeddings],
#                            name = "Embedding Layer") (input_layer)
    emb_layer = Embedding(opts.vocab_size+1, 
                            opts.embd,
                            input_length = N,
                            name = "Embedding Layer") (input_layer)
    emb_layer = Dropout(0.1, name="Dropout Embeddings")(emb_layer)
    

    LSTMEncoding = Bidirectional(LSTM(opts.lstm_units,
                                    return_sequences = True, 
                                    name="LSTM Layer")) (emb_layer)

    LSTMEncoding = Dropout(0.1, name="Dropout LSTM Layer")(LSTMEncoding)

    h_n = Lambda(get_H_n, output_shape=(k,), name = "h_n")(LSTMEncoding)

    Y = Lambda(get_H_premise, output_shape = (L, k), name = "Y")(LSTMEncoding)
    Y = TimeDistributed(Dense(k, W_regularizer = l2(0.01)))(Y)

    h_hypo = Lambda(get_H_hypothesis, output_shape = (N-L, k), name = "h_hypo")(LSTMEncoding)
    h_hypo = TimeDistributed(Dense(k, W_regularizer = l2(0.01)))(h_hypo)

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
            M = Activation('tanh')(merge([RepeatVector(L)(Wh_i), Y], mode = 'sum'))
        else:
            M = Activation('tanh')(merge([RepeatVector(L)(Wh_i), Y, RepeatVector(L)(Wr)], mode = 'sum'))

        alpha = Reshape((L, 1), input_shape=(L,))(Activation("softmax")(Flatten()(TimeDistributed(Dense(1, weights=[alpha_init_weight, alpha_init_bias]), name='alpha'+str(i))(M))))

        r = Lambda(weighted_average_pooling, output_shape=(k,), name="r"+str(i))(merge([Y, alpha], mode = 'concat', concat_axis = 2))

        if i != 1:
            r = merge([r, Tan_Wr], mode='sum')

        if i != (N-L):

            Tan_Wr = Dense(k, W_regularizer = l2(0.01),
                    activation = 'tanh',
                    name='Tan_Wr'+str(i), 
                    weights = [Tan_Wr_init_weight, Tan_Wr_init_bias])(r)
            Wr = Dense(k, W_regularizer = l2(0.01), 
                            name = 'Wr'+str(i), 
                            weights = [Wr_init_weight, Wr_init_bias])(r)


    r = Dense(k, W_regularizer = l2(0.01))(r) 
    h_n = Dense(k, W_regularizer = l2(0.01))(h_n)

    h_star = Activation('tanh')(merge([r, h_n]))

    output_layer = Dense(2, activation='softmax', name="Output Layer")(h_star)

    model = Model(input = input_layer, output = output_layer)
    model.summary()
    if opts.optimiser == 'adam':
        model.compile(loss='categorical_crossentropy', optimizer=Adam(opts.lr), metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=opts.lr, decay=opts.decay), metrics=['accuracy'])
    print "Model Compiled"

    return model

def compute_acc(X, Y, vocab, model, opts, filename=None):
    scores = model.predict(X, batch_size = options.batch_size)
    prediction = np.zeros(scores.shape)
    for i in range(scores.shape[0]):
        prediction[i][np.argmax(scores[i])] = 1.0

    assert np.array_equal(np.ones(prediction.shape[0]), np.sum(prediction, axis=1))
    
    plabels = np.argmax(prediction, axis=1)
    tlabels = np.argmax(Y, axis=1)
    a, p, r, f = getResults(tlabels, plabels)

    if filename != None:
        with open(filename, 'w') as f:
            for i in range(len(X)):
                f.write(map_to_txt(X[i],vocab)+ " : "+ str(plabels[i])+ "\n")

    return a, p, r, f

def getConfig(opts):
    conf=[opts.lstm_units,
          opts.lr,
          opts.l2,
          opts.dropout,
          opts.embd]
    return "_".join(map(lambda x: str(x), conf))

def save_model(model,wtpath,archpath):
    with open(archpath, 'w') as f:
        f.write(model.to_json())
    model.save_weights(wtpath)

def load_model(wtpath,archpath):
    with open(archpath) as f:
        model = model_from_json(f.read())
    model.load_weights(wtpath)
    return model

class WeightSharing(Callback):
    def __init__(self, shared, last_n):
        self.shared = shared
        self.last_n = last_n

    def find_layer_by_name(self, name):
        for l in self.model.layers:
            if l.name == name:
                return l

    def on_batch_end(self, batch, logs={}):
        weights = np.mean([self.find_layer_by_name(n).get_weights()[0] for n in self.shared[-self.last_n:]], axis=0)
        biases = np.mean([self.find_layer_by_name(n).get_weights()[1] for n in self.shared[-self.last_n:]], axis=0)
        for n in self.shared:
            self.find_layer_by_name(n).set_weights([weights, biases])

class WeightSave(Callback):
    def __init__(self, path, config_str):
        self.path = path
        self.config_str = config_str

    def on_epoch_end(self,epochs, logs={}):
        self.model.save_weights(self.path + self.config_str +"_"+ str(epochs) +  ".weights") 


def RTE(premise, hypothesis, vocab, model):
    labels = {0:'not_duplicate',1:'duplicate'}
    p = map_to_idx(tokenize(premise),vocab)
    h = map_to_idx(tokenize(hypothesis),vocab)
    p = pad_sequences([p], maxlen=options.xmaxlen,value=vocab["pad_tok"],padding='pre')
    h = pad_sequences([h], maxlen=options.ymaxlen,value=vocab["pad_tok"],padding='post')
    sentence = concat_in_out(p,h,vocab)
    scores = model.predict(sentence,batch_size=1)
    return labels[np.argmax(scores)]

if __name__ == "__main__":

    options=get_params()

    if options.local:
        dataPath = 'Data/'
    else:
        dataPath = '/home/cse/btech/cs1130773/Code/'

    ## Load Data
    train = [l.strip().split('\t') for l in open(dataPath+'train.txt')]
    dev = [l.strip().split('\t') for l in open(dataPath+'dev.txt')]
    test = [l.strip().split('\t') for l in open(dataPath+'test.txt')]
    vocab = eval(open(dataPath+'Vocab.txt','r').read())

    options.vocab_size = len(vocab)
    print "Vocab Size : ", len(vocab)

    X_train, Y_train, Z_train = load_data(train, vocab)
    X_dev,   Y_dev,   Z_dev   = load_data(dev,   vocab)
    X_test,  Y_test,  Z_test  = load_data(test,  vocab)
   
    setattr(K, 'params', {'xmaxlen': options.xmaxlen})
   
    XMAXLEN=options.xmaxlen
    YMAXLEN=options.ymaxlen
    X_train = pad_sequences(X_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'pre')
    X_dev   = pad_sequences(X_dev,   maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'pre')
    X_test  = pad_sequences(X_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'pre')
    Y_train = pad_sequences(Y_train, maxlen = YMAXLEN, value = vocab["pad_tok"], padding = 'post')
    Y_dev   = pad_sequences(Y_dev,   maxlen = YMAXLEN, value = vocab["pad_tok"], padding = 'post')
    Y_test  = pad_sequences(Y_test,  maxlen = YMAXLEN, value = vocab["pad_tok"], padding = 'post')
   
    net_train = concat_in_out(X_train, Y_train, vocab)
    net_dev   = concat_in_out(X_dev  , Y_dev  , vocab)
    net_test  = concat_in_out(X_test , Y_test , vocab)

    Z_train = to_categorical(Z_train, nb_classes=2)
    Z_dev   = to_categorical(Z_dev,   nb_classes=2)
    Z_test  = to_categorical(Z_test,  nb_classes=2)

    print "Training Premise Size    : ", X_train.shape
    print "Training Hypothesis Size : ", Y_train.shape

    assert net_train[0][options.xmaxlen] == vocab['delimiter']

    if options.load_save and os.path.exists(MODEL_WGHT):
        print("Loading pre-trained model from ", MODEL_WGHT)
        model = build_model(options)
        model.load_weights(MODEL_WGHT)

#        train_acc=compute_acc(net_train, Z_train, vocab, model, options)
#        dev_acc=compute_acc(net_dev, Z_dev, vocab, model, options)
#        test_acc=compute_acc(net_test, Z_test, vocab, model, options, "Test_Predictions.txt")
#        print "Training Accuracy: ", train_acc
#        print "Dev Accuracy: ", dev_acc
#        print "Testing Accuracy: ", test_acc

    else:
        print 'Building model'
        model = build_model(options)

        print 'Training New Model'
        group1 = []
        group2 = []
        group3 = []
        for i in range(1, options.ymaxlen+1):
            group1.append('Tan_Wr'+str(i))
            group2.append('Wr'+str(i))
            group3.append('alpha'+str(i))
        group3.append('alpha'+str(options.ymaxlen+1))

        ModelSaveDir = "/home/ee/btech/ee1130798/scratch/CognateModel/"

        save_weights = WeightSave(ModelSaveDir, getConfig(options))

        history = model.fit(x = net_train, 
                            y = Z_train,
                        batch_size = options.batch_size,
                        nb_epoch = options.epochs,
                        validation_data = (net_dev, Z_dev),
                        callbacks = [WeightSharing(group1, 10), WeightSharing(group2, 10), WeightSharing(group3, 10)])

        train_acc, train_pre, train_rec, train_f = compute_acc(net_train, Z_train, vocab, model, options)
        dev_acc, dev_pre, dev_rec, dev_f   = compute_acc(net_dev, Z_dev, vocab, model, options)
        test_acc, test_pre, test_rec, test_f  = compute_acc(net_test, Z_test, vocab, model, options)
        print "Training Acc: ", train_acc, "\t Precision: ", train_pre, "\t Recall: ", train_rec, "\t F-Score: ", train_f
        print "Dev Acc: ", dev_acc, "\t Precision: ", dev_pre, "\t Recall: ", dev_rec, "\t F-Score: ", dev_f
        print "Testing Acc: ", test_acc, "\t Precision: ", test_pre, "\t Recall: ", test_rec, "\t F-Score: ", test_f

        ModelSaveDir = "./"
        config_str = getConfig(options)
        MODEL_ARCH = ModelSaveDir + "ARCH_" + config_str + ".yaml"
        MODEL_WGHT = ModelSaveDir + "WGHT_" + config_str + ".weights"
        save_model(model,MODEL_WGHT,MODEL_ARCH)
