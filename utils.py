import pickle
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
from sklearn.metrics import jaccard_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import os
import time
import dataloader
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from models import *
from flags import FLAGS
import time

def print_evaluation_metrics(y_true, y_pred, model, save_file, epoch='NA'):
	print('_______________________________________________')
	print(model+' hamming loss: ', str(hamming_loss(y_true, y_pred, labels=np.unique(y_pred))))
	print(model+' jaccard similiarity accuracy: ', str(jaccard_score(y_true, y_pred, average='micro')))
	print(model+' 0/1 loss: ', str(zero_one_loss(y_true, y_pred)))
	print(model+' average precision score: '+str(average_precision_score(y_true, y_pred)))
	print(model+' macro f1 score: '+str(f1_score(y_true, y_pred, average='macro')))
	print(model+' micro f1 score: '+str(f1_score(y_true, y_pred, average='micro')))
	print('_______________________________________________')

	f=open(save_file, 'a+')
	f.write('_______________________________________________\n')
	f.write('Timestamp: '+time.ctime()+'\n')
	f.write('Total Epochs: '+str(FLAGS.end_epoch)+'\n')
	f.write('Epoch Number: '+str(epoch)+'\n')
	f.write(model+' hamming loss: '+ str(hamming_loss(y_true, y_pred, labels=np.unique(y_pred)))+'\n')
	f.write(model+' jaccard similiarity accuracy: '+ str(jaccard_score(y_true, y_pred, average='micro'))+'\n')
	f.write(model+' 0/1 loss: '+ str(zero_one_loss(y_true, y_pred))+'\n')
	f.write(model+' average precision score: '+str(average_precision_score(y_true, y_pred))+'\n')
	f.write(model+' macro f1 score: '+str(f1_score(y_true, y_pred, average='macro'))+'\n')
	f.write(model+' micro f1 score: '+str(f1_score(y_true, y_pred, average='micro'))+'\n')
	f.write('_______________________________________________')
	f.close()

def get_predictions_from_logits(random_preds_logits):
	norm_preds=torch.gt(random_preds_logits, FLAGS.prediction_threshold).long()
	return norm_preds

def transfer_weights(aux_task_network, aux_transfer_network):
	atn_dict=aux_task_network.state_dict()
	atl_dict=aux_transfer_network.state_dict()
	transfer_dict={k: v for k, v in atn_dict.items() if k in atl_dict}

	aux_transfer_network.load_state_dict(transfer_dict)
	return aux_transfer_network
