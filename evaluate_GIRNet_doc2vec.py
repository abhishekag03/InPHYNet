import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import Embedding
import keras
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import model_from_json
from dataloader_keras import *
from GIRNet import GIRNet
from sklearn.metrics import hamming_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from keras.models import load_model
import os
import tensorflow as tf
from utils import *
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
from tqdm import tqdm
from emb import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from nltk.tokenize import word_tokenize
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from dataloader_keras import *
import pickle

from flags import FLAGS

if not os.path.exists('results_girnet_doc2vec'):
		os.makedirs('results_girnet_doc2vec')

aux_data='data/aux_data_with_labels.txt'
PREPROCESSED_AUX_DATA={'x':'data/preprocessed_aux_paras.pickle', 'y':'data/preprocessed_aux_labels.pickle'}

MAX_EPOCHS=100
VEC_SIZE=500
ALPHA=0.025

paras_aux=[]
labels_aux=[]

f=open(aux_data, encoding='ISO-8859-1')
for line in f.readlines():
	first_space_pos=line.index(' ')
	first_colon=line.index(':')	
	label=line[:first_colon]
	ques=line[first_space_pos + 1: len(line) - 1]
	ques.rstrip('\n')
	label.rstrip('\n')
	paras_aux.append(ques)
	labels_aux.append(label)

train_x, train_y=FLAGS.prim_data_paras_file, FLAGS.prim_data_labels_file
paras_x=pickle.load(open(train_x, 'rb'))
paras_y=pickle.load(open(train_y, 'rb'))
paras_x, paras_y=normalize_multi_labels(paras_x, paras_y)
paras_y=binarize_multi_label(paras_y)
labels_prim=paras_y
# print(paras_x[:5], paras_y[:5])
# print(labels_aux[:5], paras_aux[:5])

model_aux=Doc2Vec.load('aux_dm.model')
model_prim=Doc2Vec.load('prim_dm.model')

seq_aux, word_index_aux=create_sequences(model_aux, paras_aux)
seq_prim, word_index_prim=create_sequences(model_prim, paras_x)

data_aux=pad_sequences(seq_aux, maxlen=200, padding="pre", truncating="post")
data_prim=pad_sequences(seq_prim, maxlen=200, padding="pre", truncating="post")

emb_matrix_aux, nb_words_aux=create_em(model_aux, word_index_aux)
emb_matrix_prim, nb_words_prim=create_em(model_prim, word_index_prim)

data_aux=np.clip(data_aux, a_min=data_aux.min(), a_max=nb_words_aux-1)

rnn_aux1=LSTM(FLAGS.num_hidden)
inp_aux1=Input(shape=(200,))
emb_aux=Embedding(nb_words_aux, 300, mask_zero=False, weights=[emb_matrix_aux], input_length=200, trainable=False)(inp_aux1)
x_a1=rnn_aux1(emb_aux)
out_aux1=Dense(FLAGS.aux_num_multi_classes, activation='softmax')(x_a1)

inp_prim=Input(shape=(200,))
emb_prim=Embedding(nb_words_prim, 300, mask_zero=False, weights=[emb_matrix_prim], input_length=200, trainable=False)(inp_prim)
# emb_prim=Embedding(phy_vocab_size, FLAGS.embedding_length, input_length=FLAGS.max_sen_len)(inp_prim)
gate_vales, prime_out, out_interleaved=GIRNet(emb_prim,  [rnn_aux1], return_sequences=False)
out_prim=Dense(FLAGS.prim_num_multi_classes, activation='softmax')(out_interleaved)

model=Model([inp_aux1, inp_prim], [out_aux1, out_prim])
model.load_weights(os.path.join('checkpoints_girnet_doc2vec', FLAGS.girnet_model_weights_save[:-3]+'_proper_split.h5'))
input_x=[]
input_y=[]

input_x.append(data_aux[:len(data_prim)])
input_x.append(data_prim)

input_y.append(labels_aux[:len(labels_prim)])
input_y.append(labels_prim)

model.summary()
model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy'], metrics = ['categorical_accuracy'], loss_weights=[0.5, 1.0])

test_input_x=[]
test_input_y=[]

test_prim_x, test_prim_y = pickle.load(open('./heldout_data/test_prim_x.pickle', 'rb')), pickle.load(open('./heldout_data/test_prim_y.pickle', 'rb'))
test_aux_x, test_aux_y=pickle.load(open('./heldout_data/test_aux_x.pickle', 'rb')), pickle.load(open('./heldout_data/test_aux_y.pickle', 'rb'))

test_input_x.append(test_aux_x[:len(test_prim_x)])
test_input_y.append(test_aux_y[:len(test_prim_y)])
test_input_x.append(test_prim_x)
test_input_y.append(test_prim_y)

predictions=model.predict(test_input_x)
prim_preds=predictions[1]
# y_true=labels_prim[1000:1800]
y_true=test_prim_y
y_pred=prim_preds

for i in range(len(y_pred)):
	for j in range(len(y_pred[i])):
		if(y_pred[i][j]>0.3):
			y_pred[i][j]=1
		else:
			y_pred[i][j]=0

print_evaluation_metrics(y_true, y_pred, 'GIRNet', os.path.join('results_girnet', FLAGS.log_file))
