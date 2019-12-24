import os
import time
import dataloader
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from utils import *
import pickle

from flags import FLAGS

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists('results_vanilla_lstm_doc2vec'):
		os.makedirs('results_vanilla_lstm_doc2vec')

print('Loading primary Physics dataset')
dataset=dataloader.PrimaryDataset()
train_iterator, val_iterator, test_iterator, vocab, word_embeddings=dataset.load_data(FLAGS.prim_data_paras_file, FLAGS.prim_data_labels_file)

embeddings_dimension=pickle.load(open(os.path.join('checkpoints_vanilla_lstm_doc2vec', FLAGS.prim_vocab_weights_save), 'rb'))
vocab_dims=embeddings_dimension[1][0]

primary_task_network=PrimaryTaskNetwork(FLAGS.batch_size, FLAGS.prim_num_multi_classes, FLAGS.num_hidden, vocab_dims, FLAGS.embedding_length, word_embeddings)

primary_state=torch.load(os.path.join('checkpoints_vanilla_lstm_doc2vec', FLAGS.primary_task_network_save))
primary_task_network.load_state_dict(primary_state['state_dict'])

primary_task_network.cuda()
primary_task_network.eval()

for idx, batch in enumerate(test_iterator):
	try:
		batch_x=batch.text
		batch_targets=batch.label

		if(batch_x.shape[1]!=FLAGS.batch_size):
			continue

		if torch.cuda.is_available():
			batch_x=batch_x.cuda()
			batch_targets=batch_targets.cuda()

		batch_preds, _, _=primary_task_network(batch_x)
		batch_preds_logits=F.softmax(batch_preds, dim=-1)
		batch_outputs=get_predictions_from_logits(batch_preds_logits)

		np_preds, np_targets=batch_outputs.cpu().detach().numpy(), batch_targets.cpu().detach().numpy()
		print_evaluation_metrics(np_targets, np_preds, 'PTN', os.path.join('results_vanilla_lstm', FLAGS.log_file))
	except RuntimeError as e:
		pass