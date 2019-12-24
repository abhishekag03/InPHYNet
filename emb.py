from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
from tqdm import tqdm

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from nltk.tokenize import word_tokenize
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from dataloader_keras import *
import pickle


vocab = Counter()
tokenizer = WordPunctTokenizer()

def process_paras_doc2vec(paras_x):
	print('Processing text dataset')
	# setup tokenizer
	list_sentences_train = paras_x #list(train_df["comment_text"].fillna("NAN_WORD").values)

	comments = process_comments(list_sentences_train, lower=True)
	return comments

def text_to_wordlist(text, lower=False):
	global vocab
	# Tokenize
	text = tokenizer.tokenize(text)
	
	# optional: lower case
	if lower:
		text = [t.lower() for t in text]
	
	# Return a list of words
	vocab.update(text)
	return text

def process_comments(list_sentences, lower=False):
	comments = []
	for text in tqdm(list_sentences):
		txt = text_to_wordlist(text, lower=lower)
		comments.append(txt)
	return comments


def create_sequences(model, paras_x):
	paras = process_paras_doc2vec(paras_x)
	word_vectors = model.wv
	print("Number of word vectors: {}".format(len(word_vectors.vocab)))
	MAX_NB_WORDS = len(word_vectors.vocab)
	MAX_SEQUENCE_LENGTH = 200
	word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(MAX_NB_WORDS))}
	sequences = [[word_index.get(t, 0) for t in para] for para in paras]
	return sequences, word_index

def create_em(model, word_index):
	WV_DIM=300
	word_vectors = model.wv
	MAX_NB_WORDS = len(word_vectors.vocab)
	nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
	wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
	for word, i in word_index.items():
		if i >= MAX_NB_WORDS:
			continue
		try:
			embedding_vector = word_vectors[word]
			if(len(word_vectors.vocab) in embedding_vector):
				continue
			wv_matrix[i] = embedding_vector
		except:
			pass 

	return wv_matrix, nb_words  


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
	label=line[:first_space_pos]
	ques=line[first_space_pos+1:len(line)-1]

	label=list(label.split(':'))[1]
	ques.rstrip('\n')
	label.rstrip('\n')

	paras_aux.append(ques)
	labels_aux.append(label)

train_x, train_y=FLAGS.prim_data_paras_file, FLAGS.prim_data_labels_file
paras_x=pickle.load(open(train_x, 'rb'))
paras_y=pickle.load(open(train_y, 'rb'))
paras_x, paras_y=normalize_multi_labels(paras_x, paras_y)
paras_y=binarize_multi_label(paras_y)

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

pickle.dump(emb_matrix_aux, open('emb_matrix_aux.pickle', 'wb'))
pickle.dump(emb_matrix_prim, open('emb_matrix_prim.pickle', 'wb'))

print("-------------")
print(emb_matrix_aux)
print(emb_matrix_prim)
print("-------------")


print(np.asarray(emb_matrix_aux).shape)
print(np.asarray(emb_matrix_prim).shape)