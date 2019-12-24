import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from itertools import cycle
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

import argparse
import getpass

parser=argparse.ArgumentParser()

# add arguments
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--aux_batch_size', type=int, default=64, help="auxiliary (TREC) data batch size")
parser.add_argument('--sst_batch_size', type=int, default=64, help="auxiliary (SST) data batch size")
parser.add_argument('--max_sen_len', type=int, default=200, help="maximum sentence length in a paragraph")
parser.add_argument('--prim_data_paras_file', type=str, default='data/paragraphs.pickle', help="file name for primary data paragraphs (in pickle format)")
parser.add_argument('--prim_data_labels_file', type=str, default='data/labels.pickle', help="file name for primary data labels (in pickle format)")
parser.add_argument('--aux_data_file', type=str, default='data/aux_data_with_labels.txt', help="file name for aux data (in txt format)")
parser.add_argument('--prim_num_multi_classes', type=int, default=9, help="number of unique multiple labels in primary dataset")
parser.add_argument('--aux_num_multi_classes', type=int, default=6, help="number of unique labels in aux (TREC) dataset")
parser.add_argument('--sst_num_multi_classes', type=int, default=4, help="number of unique labels in aux (SST) dataset")
parser.add_argument('--num_hidden', type=int, default=512, help="LSTM hidden state size")
parser.add_argument('--embedding_length', type=int, default=500, help="embedding vector dimensions")
parser.add_argument('--prediction_threshold', type=float, default=0.3, help="threshold for mult-label softmax predictions")
parser.add_argument('--num_aux', type=int, default=1, help="number of auxiliary tasks")
parser.add_argument('--lambda_aux', type=float, default=0.5, help="weight for auxiliary losses")
parser.add_argument('--max_norm', type=float, default=3, help="maximum norm cutoff for clipping LSTM gradients")
parser.add_argument('--train_test_ratio', type=float, default=0.2, help="train and test set split ratio")
parser.add_argument('--train_val_ratio', type=float, default=0.1, help="train and val set split ratio")

parser.add_argument('--lrate', type=float, default=0.0001, help="initial learning rate")
parser.add_argument('--primary_task_network_save', type=str, default='ptn', help="model save name for primary_task_network")
parser.add_argument('--aux_task_network_save', type=str, default='atn', help="model save base name for aux_task_networks")
parser.add_argument('--aux_transfer_layer_save', type=str, default='atl', help="model save for aux_transfer_layer")
parser.add_argument('--weight_alignment_layer_save', type=str, default='wal', help="model save for weight_alignment_layer")
parser.add_argument('--prim_vocab_weights_save', type=str, default='prim_vocab_weights', help="save weights for vocab embeddings for primary task")
parser.add_argument('--aux_vocab_weights_save', type=str, default='aux_vocab_weights', help="save weights for vocab embeddings for auxiliary tasks")
parser.add_argument('--girnet_model_weights_save', type=str, default='girnet_weights.h5', help="save weights for girnet model")
parser.add_argument('--girnet_model_architecture_save', type=str, default='girnet_architecture.json', help="save architecture for girnet model")

parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")

parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=50, help="flag to indicate the final epoch of training")
parser.add_argument('--is_training', type=bool, default=False, help="flag to indicate if model is in eval or train mode")

FLAGS = parser.parse_args()
