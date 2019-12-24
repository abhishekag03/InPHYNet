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
from copy import deepcopy

from flags import FLAGS

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists('checkpoints_vanilla_lstm_tfidf'):
		os.makedirs('checkpoints_vanilla_lstm_tfidf')

# Load primary and auxiliary datasets and get data loaders
print('Loading primary Physics dataset')
dataset=dataloader.PrimaryDataset()
train_iterator, test_iterator, val_iterator, vocab, word_embeddings=dataset.load_data(FLAGS.prim_data_paras_file, FLAGS.prim_data_labels_file)

# Instantiate model network
primary_task_network=PrimaryTaskNetworkTFIDF(FLAGS.batch_size, FLAGS.prim_num_multi_classes, FLAGS.num_hidden, len(vocab), FLAGS.embedding_length, word_embeddings)

# Define Loss and Optimizer
loss=torch.nn.BCELoss()
optim=torch.optim.Adam(filter(lambda p: p.requires_grad, primary_task_network.parameters()))

start_epoch=FLAGS.start_epoch
end_epoch=FLAGS.end_epoch

if FLAGS.load_saved:
	primary_state=torch.load(os.path.join('checkpoints_vanilla_lstm_tfidf', FLAGS.primary_task_network_save))
	primary_task_network.load_state_dict(primary_state['state_dict'])
	optim.load_state_dict(primary_state['optimizer'])
	start_epoch=primary_state['epoch']+1

total_loss=0
primary_task_network.cuda()
primary_task_network.train()

fixed_val_batch=next(iter(test_iterator))
while(fixed_val_batch.text.shape[1]!=FLAGS.batch_size):
	fixed_val_batch=next(iter(val_iterator))

for epoch in range(start_epoch, end_epoch+1):
	for idx, batch in enumerate(train_iterator):
		batch_x=batch.text
		batch_targets=batch.label.type(torch.cuda.FloatTensor)

		if(batch_x.shape[1]!=FLAGS.batch_size):
			continue

		if torch.cuda.is_available():
			batch_x=batch_x.cuda()
			batch_targets=batch_targets.cuda()

		optim.zero_grad()
		primary_preds, _, _=primary_task_network(batch_x)
		primary_preds_logits=torch.sigmoid(primary_preds)

		total_loss=loss(primary_preds_logits, batch_targets)
		total_loss.backward()
		optim.step()

		if(idx%20==0):
			print(f'Epoch: {epoch}, Idx: {idx}, Training Loss: {total_loss.item():.4f}')

			with torch.no_grad():			
				val_batch_x=fixed_val_batch.text
				val_batch_targets=fixed_val_batch.label

				val_preds, _, _=primary_task_network(val_batch_x.cuda())
				val_preds_logits=F.softmax(val_preds, dim=-1)
				val_batch_preds=get_predictions_from_logits(val_preds_logits)

				np_preds, np_targets=val_batch_preds.cpu().detach().numpy(), val_batch_targets.cpu().detach().numpy()
				print_evaluation_metrics(np_targets, np_preds, 'PTN', os.path.join('checkpoints_vanilla_lstm_tfidf', FLAGS.log_file))
			
	print('-------------------------------------------------------')
	if(epoch%5==0):
		state={'epoch': epoch, 'state_dict': primary_task_network.state_dict(), 'optimizer': optim.state_dict()}
		torch.save(state, os.path.join('checkpoints_vanilla_lstm_tfidf', FLAGS.primary_task_network_save))
		pickle.dump([primary_task_network.word_embeddings.weight, primary_task_network.word_embeddings.weight.shape], open(os.path.join('checkpoints_vanilla_lstm_tfidf', FLAGS.prim_vocab_weights_save), 'wb'))
