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
from sklearn.model_selection import train_test_split
from flags import FLAGS

emb_matrix_aux=pickle.load(open('emb_matrix_aux.pickle', 'rb'))
emb_matrix_prim=pickle.load(open('emb_matrix_prim.pickle', 'rb'))


if not os.path.exists('checkpoints_girnet_doc2vec'):
		os.makedirs('checkpoints_girnet_doc2vec')

def embeddings_prim(labels):
	train_x, train_y=FLAGS.prim_data_paras_file, FLAGS.prim_data_labels_file

	input_x=pickle.load(open(train_x, 'rb'))
	input_y=pickle.load(open(train_y, 'rb'))
	input_x, input_y=normalize_multi_labels(input_x, labels)
	input_y=binarize_multi_label(input_y)	
	# embedding_vectors, vocab_size=create_embedding_vectors(input_x)
	return embedding_vectors, input_y, vocab_size

def embeddings_aux(labels_):
	labelEncoder=preprocessing.LabelEncoder()
	labelEncoder.fit(labels_)
	labels_=labelEncoder.transform(labels_)
	return labels_

def load_physics_data():
	phy_x, phy_y, vocab_size=embeddings_prim()
	X_train, X_test, y_train, y_test=train_test_split(phy_x, phy_y, test_size=FLAGS.train_test_ratio)
	X_train, X_val, y_train, y_val=train_test_split(X_train, y_train, test_size=FLAGS.train_val_ratio)
	X_train=np.asarray(X_train)
	X_test=np.asarray(X_test)
	X_val=np.asarray(X_val)
	X_train=sequence.pad_sequences(X_train, maxlen=FLAGS.max_sen_len)
	X_test=sequence.pad_sequences(X_test, maxlen=FLAGS.max_sen_len)
	X_val=sequence.pad_sequences(X_val, maxlen=FLAGS.max_sen_len)
	return X_train, X_test, X_val, np.asarray(y_train), np.asarray(y_test), np.asarray(y_val), vocab_size

def load_trec_data():
	trec_x, trec_y, vocab_size=embeddings_aux()
	X_train, X_test, y_train, y_test=train_test_split(trec_x, trec_y, test_size=FLAGS.train_test_ratio)
	X_train, X_val, y_train, y_val=train_test_split(X_train, y_train, test_size=FLAGS.train_val_ratio)
	X_train=np.asarray(X_train)
	X_test=np.asarray(X_test)
	X_val=np.asarray(X_val)
	X_train=sequence.pad_sequences(X_train, maxlen=FLAGS.max_sen_len)
	X_test=sequence.pad_sequences(X_test, maxlen=FLAGS.max_sen_len)
	X_val=sequence.pad_sequences(X_val, maxlen=FLAGS.max_sen_len)
	return X_train, X_test, X_val, np.asarray(y_train), np.asarray(y_test), np.asarray(y_val), vocab_size

def encode_one_hot(labels, num_classes=FLAGS.aux_num_multi_classes):
	enc_labels=to_categorical(labels)
	return enc_labels

def build_model(emb_matrix_aux, emb_matrix_prim, nb_words_aux, nb_words_prim, data_aux, data_prim, labels_aux, labels_prim):

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
	input_x=[]
	input_y=[]
	train_prim_x, test_prim_x, train_prim_y, test_prim_y=train_test_split(data_prim, labels_prim, test_size=0.2, random_state=42)
	train_aux_x, test_aux_x, train_aux_y, test_aux_y=train_test_split(data_aux, labels_aux, test_size=0.2, random_state=42)

	pickle.dump(test_prim_x, open('./heldout_data/test_prim_x.pickle', 'wb'))
	pickle.dump(test_prim_y, open('./heldout_data/test_prim_y.pickle', 'wb'))
	pickle.dump(test_aux_x, open('./heldout_data/test_aux_x.pickle', 'wb'))
	pickle.dump(test_aux_y, open('./heldout_data/test_aux_y.pickle', 'wb'))

	input_x.append(train_aux_x[:len(train_prim_x)])
	input_x.append(train_prim_x)

	input_y.append(train_aux_y[:len(train_prim_y)])
	input_y.append(train_prim_y)

	model.summary()
	model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy'], metrics = ['categorical_accuracy'], loss_weights=[0.5, 1.0])
	history=model.fit(input_x, input_y, validation_split=0.2, epochs=FLAGS.end_epoch, verbose=1, shuffle=True)

	model.save_weights(os.path.join('checkpoints_girnet_doc2vec', FLAGS.girnet_model_weights_save[:-3]+'_proper_split.h5'))	
	with open(os.path.join('checkpoints_girnet_doc2vec', FLAGS.girnet_model_architecture_save), 'w') as f:
		f.write(model.to_json())

	test_input_x=[]
	test_input_y=[]

	test_input_x.append(test_aux_x[:len(test_prim_x)])
	test_input_y.append(test_aux_y[:len(test_prim_y)])
	test_input_x.append(test_prim_x)
	test_input_y.append(test_prim_y)

	predictions=model.predict(test_input_x)
	prim_preds=predictions[1]

	y_true=test_prim_y
	y_pred=prim_preds

	for i in range(len(y_pred)):
		for j in range(len(y_pred[i])):
			if(y_pred[i][j]>0.6):
				y_pred[i][j]=1
			else:
				y_pred[i][j]=0

	print_evaluation_metrics(y_true, y_pred, 'GIRNet', os.path.join('checkpoints_girnet_doc2vec', FLAGS.log_file))

if __name__ == '__main__':


	aux_data='data/aux_data_with_labels.txt'

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

	labels_aux=embeddings_aux(labels_aux)
	labels_aux=encode_one_hot(labels_aux)
	model_aux=Doc2Vec.load('aux_dm.model')
	seq_aux, word_index_aux=create_sequences(model_aux, paras_aux)
	data_aux=pad_sequences(seq_aux, maxlen=200, padding="pre", truncating="post")
	emb_matrix_aux, nb_words_aux=create_em(model_aux, word_index_aux)
	data_aux=np.clip(data_aux, a_min=data_aux.min(), a_max=nb_words_aux-1)

	#Primary data
	prim_data_x, prim_data_y=pickle.load(open(FLAGS.prim_data_paras_file, 'rb')), pickle.load(open(FLAGS.prim_data_labels_file, 'rb'))
	prim_data_x, prim_data_y=normalize_multi_labels(prim_data_x, prim_data_y)
	prim_data_y=binarize_multi_label(prim_data_y)
	model_prim=Doc2Vec.load('prim_dm.model')
	seq_prim, word_index_prim=create_sequences(model_prim, prim_data_x)
	data_prim=pad_sequences(seq_prim, maxlen=200, padding="pre", truncating="post")
	emb_matrix_prim, nb_words_prim=create_em(model_prim, word_index_prim)

	build_model(emb_matrix_aux, emb_matrix_prim, nb_words_aux, nb_words_prim, data_aux, data_prim, labels_aux, prim_data_y)
