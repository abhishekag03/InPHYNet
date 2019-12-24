import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import pickle
import spacy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec
from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords

from flags import FLAGS

LABELMAP={'def':0, 'cau':1, 'exa':2, 'rea':3, 'pro':4, 'typ':5, 'eff':6, 'for':7, 'equ':7, 'rel':8}

class PrimaryDataset(object):
	def __init__(self):
		self.train_iterator=None
		self.test_iterator=None
		self.val_iterator=None
		self.vocab=[]
		self.word_embeddings={}

	def normalize_multi_labels(self, input_paras, input_labels):

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

	def binarize_multi_label(self, label_list):
		mlb=MultiLabelBinarizer()
		mlb.fit(label_list)
		return mlb.transform(label_list)
	
	def load_data(self, train_x, train_y, test_x=None, test_y=None):
		NLP=spacy.load('en_core_web_sm')
		tokenizer=lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
		
		TEXT=data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=FLAGS.max_sen_len)
		LABEL=data.Field(sequential=False, use_vocab=False)
		datafields=[("text",TEXT),("label",LABEL)]
		
		input_x=pickle.load(open(train_x, 'rb'))
		input_y=pickle.load(open(train_y, 'rb'))
		input_x, input_y=self.normalize_multi_labels(input_x, input_y)
		input_y=self.binarize_multi_label(input_y)

		input_data=[]
		for i in range(len(input_x)):
			input_data.append([input_x[i], list(input_y[i])])

		train_examples=[data.Example.fromlist(i, datafields) for i in input_data]
		train_data=data.Dataset(train_examples, datafields)

		test_data, train_data=train_data.split(split_ratio=FLAGS.train_test_ratio)
		val_data, train_data=train_data.split(split_ratio=FLAGS.train_val_ratio)

		TEXT.build_vocab(train_data)
		self.vocab=TEXT.vocab
		
		self.train_iterator=data.BucketIterator(
			(train_data),
			batch_size=FLAGS.batch_size,
			sort_key=lambda x: len(x.text),
			repeat=False,
			shuffle=True)
		
		self.test_iterator=data.BucketIterator(
			(test_data),
			batch_size=FLAGS.batch_size,
			sort_key=lambda x: len(x.text),
			repeat=False,
			shuffle=False)
		
		self.val_iterator=data.BucketIterator(
			(val_data),
			batch_size=FLAGS.batch_size,
			sort_key=lambda x: len(x.text),
			repeat=False,
			shuffle=False)

		print ("Loaded {} training examples".format(len(train_data)))
		print ("Loaded {} test examples".format(len(test_data)))
		print ("Loaded {} val examples".format(len(val_data)))

		return self.train_iterator, self.test_iterator, self.val_iterator, self.vocab, self.word_embeddings

class AuxiliaryDataset(object):
	def __init__(self):
		self.train_iterator=None
		self.test_iterator=None
		self.val_iterator=None
		self.vocab=[]
		self.label_vocab=[]

	def load_data(self):
		NLP=spacy.load('en_core_web_sm')
		tokenizer=lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

		TEXT=data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=FLAGS.max_sen_len)
		LABEL=data.Field(sequential=False, use_vocab=True)

		train_data, _=datasets.TREC.splits(TEXT, LABEL)

		test_data, train_data=train_data.split(split_ratio=FLAGS.train_test_ratio)
		val_data, train_data=train_data.split(split_ratio=FLAGS.train_val_ratio)

		TEXT.build_vocab(train_data)
		LABEL.build_vocab(train_data)
		self.vocab=TEXT.vocab
		self.label_vocab=LABEL.vocab

		self.train_iterator=data.BucketIterator(
			(train_data),
			batch_size=FLAGS.aux_batch_size,
			sort_key=lambda x: len(x.text),
			repeat=False,
			shuffle=True)
		
		self.test_iterator=data.BucketIterator(
			(test_data),
			batch_size=FLAGS.aux_batch_size,
			sort_key=lambda x: len(x.text),
			repeat=False,
			shuffle=False)

		self.val_iterator=data.BucketIterator(
			(val_data),
			batch_size=FLAGS.aux_batch_size,
			sort_key=lambda x: len(x.text),
			repeat=False,
			shuffle=False)

		print ("Loaded {} training examples".format(len(train_data)))
		print ("Loaded {} test examples".format(len(test_data)))
		print ("Loaded {} val examples".format(len(val_data)))

		return self.train_iterator, self.test_iterator, self.val_iterator, self.vocab

class SSTDataset(object):
	def __init__(self):
		self.train_iterator=None
		self.test_iterator=None
		self.val_iterator=None
		self.vocab=[]
		self.label_vocab=[]

	def load_data(self):
		NLP=spacy.load('en_core_web_sm')
		tokenizer=lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

		TEXT=data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=FLAGS.max_sen_len)
		LABEL=data.Field(sequential=False, use_vocab=True)

		train_data, _, _=datasets.SST.splits(TEXT, LABEL)

		test_data, train_data=train_data.split(split_ratio=FLAGS.train_test_ratio)
		val_data, train_data=train_data.split(split_ratio=FLAGS.train_val_ratio)

		TEXT.build_vocab(train_data)
		LABEL.build_vocab(train_data)
		self.vocab=TEXT.vocab
		self.label_vocab=LABEL.vocab

		self.train_iterator=data.BucketIterator(
			(train_data),
			batch_size=FLAGS.sst_batch_size,
			sort_key=lambda x: len(x.text),
			repeat=False,
			shuffle=True)
		
		self.test_iterator=data.BucketIterator(
			(test_data),
			batch_size=FLAGS.sst_batch_size,
			sort_key=lambda x: len(x.text),
			repeat=False,
			shuffle=False)

		self.val_iterator=data.BucketIterator(
			(val_data),
			batch_size=FLAGS.sst_batch_size,
			sort_key=lambda x: len(x.text),
			repeat=False,
			shuffle=False)

		print ("Loaded {} training examples".format(len(train_data)))
		print ("Loaded {} test examples".format(len(test_data)))
		print ("Loaded {} val examples".format(len(val_data)))

		return self.train_iterator, self.test_iterator, self.val_iterator, self.vocab

if __name__ == '__main__':
	print('Physics data')
	dataset=PrimaryDataset()
	train_iterator, test_iterator, val_iterator, vocab, word_embeddings=dataset.load_data(FLAGS.prim_data_paras_file, FLAGS.prim_data_labels_file)

	print('TREC data')
	dataset=AuxiliaryDataset()
	train_iterator, test_iterator, val_iterator, vocab=dataset.load_data()

	print('SST data')
	dataset=SSTDataset()
	train_iterator, test_iterator, val_iterator, vocab=dataset.load_data()