import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from gensim.models import KeyedVectors
import pickle
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from flags import FLAGS

import warnings
warnings.filterwarnings("ignore")

LABELMAP={'def':0, 'cau':1, 'exa':2, 'rea':3, 'pro':4, 'typ':5, 'eff':6, 'for':7, 'equ':7, 'rel':8}

def create_embedding_vectors(paras):
	tokenizer=Tokenizer(num_words=10000)
	tokenizer.fit_on_texts(paras)
	vectors=tokenizer.texts_to_sequences(paras)
	vocab_size=len(tokenizer.word_index)+1
	return vectors, vocab_size

def normalize_multi_labels(input_paras, input_labels):
	stop=set(stopwords.words('english'))

	# 1. Definition
	# 2. Causes
	# 3. Examples
	# 4. Reasoning
	# 5. Property
	# 6. Types
	# 7. Effects
	# 8. Formula/Equation
	# 9. Relation

	output_labels=[]
	output_paras=[]
	i=0
	for label_array in input_labels:
		inter_labels=[]
		label_array=label_array.lower()
		labels=label_array.split(',')
		flag=0
		for label in labels:
			label=label.strip()
			if(label.isalpha()):
				try:
					inter_labels.append(LABELMAP[label[:3]])
				except Exception as e:
					flag=1
					break
		if(flag==1):
			pass
		else:
			text=input_paras[i]
			text=text.lower()
			text=' '.join([word for word in text.split() if word not in stop])
			output_paras.append(text)
			output_labels.append(inter_labels)
		i+=1
	return output_paras, output_labels

def binarize_multi_label(label_list):
	mlb=MultiLabelBinarizer()
	mlb.fit(label_list)
	return mlb.transform(label_list)

def embeddings_prim():
	train_x, train_y=FLAGS.prim_data_paras_file, FLAGS.prim_data_labels_file

	input_x=pickle.load(open(train_x, 'rb'))
	input_y=pickle.load(open(train_y, 'rb'))
	input_x, input_y=normalize_multi_labels(input_x, input_y)
	input_y=binarize_multi_label(input_y)

	
	embedding_vectors, vocab_size=create_embedding_vectors(input_x)
	return embedding_vectors, input_y, vocab_size

def embeddings_aux():
	f=open(FLAGS.aux_data_file, encoding='ISO-8859-1')
	paras=[]
	labels=[]
	for line in f.readlines():
		first_space_pos=line.index(' ')
		first_colon=line.index(':')	
		label=line[:first_colon]
		ques=line[first_space_pos + 1: len(line) - 1]
		ques.rstrip('\n')
		label.rstrip('\n')
		paras.append(ques)
		labels.append(label)

	labelEncoder=preprocessing.LabelEncoder()
	labelEncoder.fit(labels)
	labels=labelEncoder.transform(labels)
	embedding_vectors, vocab_size=create_embedding_vectors(paras)
	return embedding_vectors, labels, vocab_size

if __name__ == '__main__':
	print('Primary Physics')
	embedding_vectors, labels, vocab_size=embeddings_prim()
	print(np.asarray(embedding_vectors).shape, np.asarray(labels).shape, vocab_size)

	print('Auxiliary TREC')
	embedding_vectors, labels, vocab_size=embeddings_aux()
	print(np.asarray(embedding_vectors).shape, np.asarray(labels).shape, vocab_size)