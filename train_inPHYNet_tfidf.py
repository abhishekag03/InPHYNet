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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from utils import *
from copy import deepcopy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from flags import FLAGS

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists('checkpoints_inphynet_tfidf'):
		os.makedirs('checkpoints_inphynet_tfidf')

model_tf_prim=Doc2Vec.load('prim_dm_INPHYNET.model')

# Load primary and auxiliary datasets and get data loaders
print('Loading primary Physics dataset')
prim_dataset=dataloader.PrimaryDataset()
prim_train_iterator, prim_test_iterator, prim_val_iterator, prim_vocab, prim_word_embeddings=prim_dataset.load_data(FLAGS.prim_data_paras_file, FLAGS.prim_data_labels_file)

print('Loading auxiliary TREC dataset')
aux_dataset=dataloader.AuxiliaryDataset()
aux_train_iterator, aux_test_iterator, aux_val_iterator, aux_vocab=aux_dataset.load_data()

# Instantiate model networks
primary_task_network=PrimaryTaskNetworkTFIDF(FLAGS.batch_size, FLAGS.prim_num_multi_classes, FLAGS.num_hidden, len(prim_vocab), FLAGS.embedding_length, model_tf_prim, None)
aux_task_network=AuxTaskNetwork(FLAGS.aux_batch_size, FLAGS.aux_num_multi_classes+1, FLAGS.num_hidden, len(aux_vocab), FLAGS.embedding_length)
aux_transfer_network=AuxTransferNetwork(FLAGS.batch_size, FLAGS.num_hidden, FLAGS.embedding_length)
weight_alignment_network=WeightAlignmentLayer(FLAGS.batch_size, FLAGS.num_aux, FLAGS.num_hidden, FLAGS.prim_num_multi_classes)

#Define Loss and Optimizers
bce_loss=torch.nn.BCELoss()
cross_entropy_loss=nn.CrossEntropyLoss()
nll_loss=nn.NLLLoss()

ptn_params=list(filter(lambda p: p.requires_grad, primary_task_network.parameters()))
atn_params=list(filter(lambda p: p.requires_grad, aux_task_network.parameters()))
atl_params=list(filter(lambda p: p.requires_grad, aux_transfer_network.parameters()))
wal_params=list(filter(lambda p: p.requires_grad, weight_alignment_network.parameters()))

ptn_optimizer=torch.optim.Adam(ptn_params)
atn_optimizer=torch.optim.Adam(atn_params)
atl_optimizer=torch.optim.Adam(atl_params)
wal_optimizer=torch.optim.Adam(wal_params)
# ptn_optimizer=torch.optim.Adamax(ptn_params)
# atn_optimizer=torch.optim.Adamax(atn_params)
# atl_optimizer=torch.optim.Adamax(atl_params)
# wal_optimizer=torch.optim.Adamax(wal_params)

start_epoch=FLAGS.start_epoch
end_epoch=FLAGS.end_epoch

if FLAGS.load_saved:
	primary_state=torch.load(os.path.join('checkpoints_inphynet_tfidf', FLAGS.primary_task_network_save))
	primary_task_network.load_state_dict(primary_state['state_dict'])
	ptn_optimizer.load_state_dict(primary_state['optimizer'])
	start_epoch=primary_state['epoch']+1

	aux_state=torch.load(os.path.join('checkpoints_inphynet_tfidf', FLAGS.aux_task_network_save))
	aux_task_network.load_state_dict(aux_state['state_dict'])
	atn_optimizer.load_state_dict(aux_state['optimizer'])

	aux_t_state=torch.load(os.path.join('checkpoints_inphynet_tfidf', FLAGS.aux_transfer_layer_save))
	aux_transfer_network.load_state_dict(aux_t_state['state_dict'])
	atl_optimizer.load_state_dict(aux_t_state['optimizer'])

	wal_state=torch.load(os.path.join('checkpoints_inphynet_tfidf', FLAGS.weight_alignment_layer_save))
	weight_alignment_network.load_state_dict(wal_state['state_dict'])
	wal_optimizer.load_state_dict(wal_state['optimizer'])

total_loss=0
aux_loss=0
prim_loss=0
primary_task_network.cuda()
primary_task_network.train()
aux_task_network.cuda()
aux_task_network.train()
aux_transfer_network.cuda()
aux_transfer_network.train()
weight_alignment_network.cuda()
weight_alignment_network.train()

composite_hidden_state=Variable(torch.zeros(1, FLAGS.batch_size, FLAGS.num_hidden))
composite_cell_state=Variable(torch.zeros(1, FLAGS.batch_size, FLAGS.num_hidden))

if torch.cuda.is_available():
	composite_hidden_state=composite_hidden_state.cuda()
	composite_cell_state=composite_cell_state.cuda()

fixed_prim_val_batch=next(iter(prim_val_iterator))
while(fixed_prim_val_batch.text.shape[1]!=FLAGS.batch_size):
	fixed_prim_val_batch=next(iter(prim_val_iterator))

fixed_aux_val_batch=next(iter(aux_val_iterator))
while(fixed_aux_val_batch.text.shape[1]!=FLAGS.aux_batch_size):
	fixed_aux_val_batch=next(iter(aux_val_iterator))

	
aux_loss_history = []
prim_loss_history = []
total_loss_history = []

for epoch in range(start_epoch, end_epoch+1):
	epoch_aux_loss = 0
	epoch_prim_loss = 0
	epoch_total_loss = 0
	count = 0
	for idx, (prim_batch, aux_batch) in enumerate(zip(prim_train_iterator, aux_train_iterator)):
		count += 1
		prim_batch_x=prim_batch.text
		prim_batch_targets=prim_batch.label.type(torch.cuda.FloatTensor)
		# prim_batch_targets=prim_batch.label.type(torch.FloatTensor)

		aux_batch_x=aux_batch.text
		aux_batch_targets=aux_batch.label.type(torch.cuda.LongTensor)
		# aux_batch_targets=aux_batch.label.type(torch.LongTensor)

		if(prim_batch_x.shape[1]!=FLAGS.batch_size):
			continue

		if(aux_batch_x.shape[1]!=FLAGS.aux_batch_size):
			continue

		if torch.cuda.is_available():
			prim_batch_x=prim_batch_x.cuda()
			prim_batch_targets=prim_batch_targets.cuda()
			aux_batch_x=aux_batch_x.cuda()
			aux_batch_targets=aux_batch_targets.cuda()

		ptn_optimizer.zero_grad()
		atn_optimizer.zero_grad()
		wal_optimizer.zero_grad()

		aux_preds, _, _=aux_task_network(aux_batch_x)
		log_softmax_layer=nn.LogSoftmax(dim=0)
		aux_preds_logits=log_softmax_layer(aux_preds)
		aux_loss=nll_loss(aux_preds_logits, aux_batch_targets)
		# aux_loss=cross_entropy_loss(aux_preds_logits, aux_batch_targets)

		atn_copy=AuxTaskNetwork(FLAGS.aux_batch_size, FLAGS.aux_num_multi_classes+1, FLAGS.num_hidden, len(aux_vocab), FLAGS.embedding_length)
		atn_copy.load_state_dict(aux_task_network.state_dict()) 

		atl_copy=AuxTransferNetwork(FLAGS.batch_size, FLAGS.num_hidden, FLAGS.embedding_length)
		atl_copy.load_state_dict(aux_transfer_network.state_dict())

		if torch.cuda.is_available():
			atn_copy.cuda()
			atl_copy.cuda()

		aux_transfer_network=transfer_weights(atn_copy, atl_copy)
		aux_hidden_state, aux_cell_state=aux_transfer_network(composite_cell_state, composite_hidden_state, FLAGS.batch_size)
		primary_preds, primary_h, primary_c=primary_task_network(prim_batch_x)
		composite_preds, composite_cell_state, composite_hidden_state=weight_alignment_network(aux_cell_state, aux_hidden_state, composite_hidden_state, composite_cell_state, primary_h)
		composite_preds_logits=torch.sigmoid(composite_preds)
		primary_preds_logits=torch.sigmoid(primary_preds)

		prim_loss=bce_loss(primary_preds_logits, prim_batch_targets)

		prim_loss.backward(retain_graph=True)
		# nn.utils.clip_grad_norm(ptn_params, max_norm=15)
		nn.utils.clip_grad_norm(ptn_params, max_norm=10)
		ptn_optimizer.step()

		aux_loss.backward(retain_graph=True)
		# nn.utils.clip_grad_norm(atn_params, max_norm=FLAGS.max_norm)
		nn.utils.clip_grad_norm(atn_params, max_norm=10)

		atn_optimizer.step()

		total_loss=FLAGS.lambda_aux*aux_loss+prim_loss

		# total_loss.backward()
		wal_optimizer.step()

		composite_cell_state=composite_cell_state.detach()
		composite_hidden_state=composite_hidden_state.detach()

		epoch_aux_loss += aux_loss.item()
		epoch_prim_loss += prim_loss.item()
		epoch_total_loss += total_loss.item()

		if(idx%20==0):
			# print(f'Epoch: {epoch}, Idx: {idx}, Training Loss: {total_loss.item():.4f}, Primary Loss: {prim_loss.item():.4f}, Auxiliary Loss: {aux_loss.item():.4f}')
			# print(f'Epoch: {epoch}, Idx: {idx} Primary Loss: {prim_loss.item():.4f}')
			print('Epoch: '+str(epoch)+', Idx: '+str(idx)+' Primary Loss: '+str(prim_loss.item()))

			with torch.no_grad():

				prim_val_batch_x=fixed_prim_val_batch.text
				prim_val_batch_targets=fixed_prim_val_batch.label

				prim_val_preds, _, _=primary_task_network(prim_val_batch_x.cuda())
				# prim_val_preds, _, _=primary_task_network(prim_val_batch_x)
				prim_val_preds_logits=F.softmax(prim_val_preds, dim=-1)
				prim_val_batch_preds=get_predictions_from_logits(prim_val_preds_logits)

				prim_np_preds, prim_np_targets=prim_val_batch_preds.cpu().detach().numpy(), prim_val_batch_targets.cpu().detach().numpy()
				print_evaluation_metrics(prim_np_targets, prim_np_preds, 'inphynet', os.path.join('checkpoints_inphynet_tfidf', FLAGS.log_file), epoch)
			
				# aux_val_batch_x=fixed_aux_val_batch.text
				# aux_val_batch_targets=fixed_aux_val_batch.label

				# aux_val_preds, _, _=aux_task_network(aux_val_batch_x.cuda())
				# aux_val_preds_logits=F.softmax(aux_val_preds, dim=-1)
				# aux_val_batch_preds=torch.argmax(aux_val_preds_logits, dim=-1)

				# aux_np_preds, aux_np_targets=aux_val_batch_preds.cpu().detach().numpy(), aux_val_batch_targets.cpu().detach().numpy()
				# print('Accuracy score: ', str(accuracy_score(aux_np_preds, aux_np_targets)))

	print('-------------------------------------------------------')
	aux_loss_history.append(epoch_aux_loss/count)
	prim_loss_history.append(epoch_prim_loss/count)
	total_loss_history.append(epoch_total_loss/count)
	print("Losses: ", epoch_aux_loss/count, epoch_prim_loss/count, epoch_total_loss/count)
	if(epoch%5==0):
		prim_state={'epoch': epoch, 'state_dict': primary_task_network.state_dict(), 'optimizer': ptn_optimizer.state_dict()}
		torch.save(prim_state, os.path.join('checkpoints_inphynet_tfidf', FLAGS.primary_task_network_save))

		aux_state={'epoch': epoch, 'state_dict': aux_task_network.state_dict(), 'optimizer': atn_optimizer.state_dict()}
		torch.save(aux_state, os.path.join('checkpoints_inphynet_tfidf', FLAGS.aux_task_network_save))

		wal_state={'epoch': epoch, 'state_dict': weight_alignment_network.state_dict(), 'optimizer': wal_optimizer.state_dict()}
		torch.save(wal_state, os.path.join('checkpoints_inphynet_tfidf', FLAGS.weight_alignment_layer_save))

		pickle.dump([primary_task_network.word_embeddings.weight, primary_task_network.word_embeddings.weight.shape], open(os.path.join('checkpoints_inphynet_tfidf', FLAGS.prim_vocab_weights_save), 'wb'))
		pickle.dump([aux_task_network.word_embeddings.weight, aux_task_network.word_embeddings.weight.shape], open(os.path.join('checkpoints_inphynet_tfidf', FLAGS.aux_vocab_weights_save), 'wb'))


pickle.dump(aux_loss_history, open('inphynet_aux_loss_tfidf.pickle', 'wb'))
pickle.dump(prim_loss_history, open('inphynet_prim_loss_tfidf.pickle', 'wb'))
pickle.dump(total_loss_history, open('inphynet_total_loss_tfidf.pickle', 'wb'))
