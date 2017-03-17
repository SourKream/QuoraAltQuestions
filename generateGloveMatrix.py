import numpy as np
import pickle

vocab = eval(open('Data/Vocab.txt','r').read())

glove = {}
for line in open('glove.840B.300d.txt', 'r'):
	line = line.strip().split()
	if line[0] in vocab:
		glove[line[0]] = np.matrix([float(x) for x in line[1:]])

# s = np.sqrt(2. / (len(vocab) + 1 + 300))
# E = np.random.normal(0, s, (len(vocab) + 1, 300))

E = np.random.normal(0, 1, (len(vocab) + 1, 300))

for word in vocab:
	if word in glove:
		E[vocab[word], :] = glove[word]

with open('GloveEmbeddingMatrix.pkl','w') as f:
	pickle.dump(E, f)

