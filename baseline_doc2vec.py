import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import scipy as sp
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec
from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import nltk
from nltk.corpus import stopwords
from emb import *
from dataloader_keras import *
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from flags import FLAGS
from nltk.tokenize import word_tokenize

LABELMAP={'def':0, 'cau':1, 'exa':2, 'rea':3, 'pro':4, 'typ':5, 'eff':6, 'for':7, 'rel':8, 'equ':7}
output_labels_file='data/preprocessed_labels.pickle'
output_paras_file='data/preprocessed_paras.pickle'

def normalizeLabels(inputLabels, paras):

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

	# print(inputLabels)

	# print(len(inputLabels))

	outputLabels=[]
	outputParas=[]

	i=0

	for label_array in inputLabels:

		inter_labels=[]

		label_array=label_array.lower()
		labels=label_array.split(',')

		# print(labels)

		flag=0

		for label in labels:
			label=label.strip()
			# print(label[:3])
			if(label.isalpha()):
				try:
					inter_labels.append(LABELMAP[label[:3]])
				except Exception as e:
					# print(label)
					# del paras[i]
					flag=1
					# print(paras[i])
					break

		if(flag==1):
			pass
		else:
			text=paras[i]
			text=text.lower()
			text=' '.join([word for word in text.split() if word not in stop])
			# outputParas.append(paras[i])
			outputParas.append(text)
			outputLabels.append(inter_labels)

		# print(inter_labels)

		i+=1

	# print(len(outputParas), len(outputLabels), 'sdbbfgs')

	return outputLabels, outputParas

def one_hot_labels(labels):
	multiLabelBinarizer=MultiLabelBinarizer()
	# print(list(multiLabelBinarizer.fit_transform(labels))[:5])
	return list(multiLabelBinarizer.fit_transform(labels)), len(multiLabelBinarizer.classes_)

def preprocess_labels(paras, labels):
	output_labels, output_paras=normalize_multi_labels(labels, paras)
	pickle.dump(output_labels, open(output_labels_file, 'wb'))
	pickle.dump(output_paras, open(output_paras_file, 'wb'))

def print_evaluation_metrics(y_true, y_pred, model):
	print('_______________________________________________')
	print(model+' hamming loss: ', str(hamming_loss(y_true, y_pred)))
	print(model+' jaccard similiarity accuracy: ', str(jaccard_similarity_score(y_true, y_pred)))
	print(model+' 0/1 loss: ', str(zero_one_loss(y_true, y_pred)))
	print(model+' average precision score: '+str(average_precision_score(y_true, y_pred)))
	print(model+' macro f1 score: '+str(f1_score(y_true, y_pred, average='macro')))
	print(model+' micro f1 score: '+str(f1_score(y_true, y_pred, average='micro')))
	print('_______________________________________________')


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

paras_x, paras_aux=np.asarray(paras_x), np.asarray(paras_aux)

print(paras_x.shape)
print(paras_y.shape)

model_aux=Doc2Vec.load('aux_dm.model')
model_prim=Doc2Vec.load('prim_dm.model')


training_x=[]
training_y=[]

for i in range(len(paras_x)):
	training_x.append(model_prim.infer_vector(word_tokenize(paras_x[i].lower())))

pca = PCA(n_components=50)
training_x=pca.fit_transform(training_x)
training_y=paras_y

print(training_x.shape)

train_x, test_x, train_y, test_y=train_test_split(training_x, training_y, test_size=0.3, shuffle=True)
train_x, test_x, train_y, test_y=np.asarray(train_x), np.asarray(test_x), np.asarray(train_y), np.asarray(test_y)

classifier=BinaryRelevance(GaussianNB())
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'BinaryRelevance GaussianNB')

classifier=BinaryRelevance(DecisionTreeClassifier())
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'BinaryRelevance DecisionTreeClassifier')

classifier=BinaryRelevance(RandomForestClassifier())
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'BinaryRelevance RandomForestClassifier')

# classifier=BinaryRelevance(MultinomialNB())
# classifier.fit(train_x, train_y)
# print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'BinaryRelevance MultinomialNB')

classifier=BinaryRelevance(MLPClassifier(activation='logistic', hidden_layer_sizes=(100)))
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'BinaryRelevance MLP logistic')

classifier=BinaryRelevance(MLPClassifier(activation='relu', hidden_layer_sizes=(100)))
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'BinaryRelevance MLP relu')

# # #############################################################################

classifier=ClassifierChain(GaussianNB())
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'ClassifierChain GaussianNB')

classifier=ClassifierChain(DecisionTreeClassifier())
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'ClassifierChain DecisionTreeClassifier')

# classifier=ClassifierChain(MultinomialNB())
# classifier.fit(train_x, train_y)
# print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'ClassifierChain MultinomialNB')

classifier=ClassifierChain(MLPClassifier(activation='logistic', hidden_layer_sizes=(100)))
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'ClassifierChain MLP logistic')

classifier=ClassifierChain(MLPClassifier(activation='relu', hidden_layer_sizes=(100)))
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'ClassifierChain MLP relu')

# # ###############################################################################

classifier=LabelPowerset(GaussianNB())
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'LabelPowerset GaussianNB')

classifier=LabelPowerset(DecisionTreeClassifier())
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'LabelPowerset DecisionTreeClassifier')

classifier=LabelPowerset(RandomForestClassifier())
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'LabelPowerset RandomForestClassifier')

# classifier=LabelPowerset(MultinomialNB())
# classifier.fit(train_x, train_y)
# print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'LabelPowerset MultinomialNB')

classifier=LabelPowerset(MLPClassifier(activation='logistic', hidden_layer_sizes=(100)))
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'LabelPowerset MLP logistic')

classifier=LabelPowerset(MLPClassifier(activation='relu', hidden_layer_sizes=(100)))
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'LabelPowerset MLP relu')

###############################################################################

classifier=MLkNN(k=20)
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'MLkNN 20')

classifier=MLkNN(k=10)
classifier.fit(train_x, train_y)
print_evaluation_metrics(test_y, classifier.predict(test_x).toarray(), 'MLkNN 10')