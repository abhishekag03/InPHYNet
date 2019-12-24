import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import pickle

from flags import FLAGS

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from nltk.tokenize import word_tokenize

model_aux=Doc2Vec.load('aux_dm.model')
model_prim=Doc2Vec.load('prim_dm.model')

class PrimaryTaskNetwork(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights=None):
		super(PrimaryTaskNetwork, self).__init__()
		
		self.batch_size=batch_size
		self.output_size=output_size
		self.hidden_size=hidden_size
		self.vocab_size=len(model_prim.wv.vocab)
		self.embedding_length=embedding_length

		self.word_embeddings=nn.Embedding.from_pretrained(torch.FloatTensor(model_prim.wv.vectors))

		# self.word_embeddings=nn.Embedding(vocab_size, embedding_length)
		
		if(weights is None):
			weights=[]

		if(len(weights)!=0):
			self.word_embeddings.weight=nn.Parameter(weights, requires_grad=False)

		self.lstm=nn.LSTM(embedding_length, hidden_size)
		self.label=nn.Linear(hidden_size, output_size)

	def embedding_(self, input_sentence):
		return self.word_embeddings(input_sentence)
		
	def forward(self, input_sentence, batch_size=None):
		input_=self.embedding_(input_sentence)

		if batch_size is None:
			h_0=Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) 
			c_0=Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
		else:
			h_0=Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			c_0=Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		output, (final_hidden_state, final_cell_state)=self.lstm(input_, (h_0, c_0))

		# final_output=self.label(final_hidden_state[-1])		
		final_output=self.label(output[-1])		
		return final_output, final_hidden_state, final_cell_state


class PrimaryTaskNetworkTFIDF(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, emb_tf, weights=None):
		super(PrimaryTaskNetworkTFIDF, self).__init__()
		
		self.batch_size=batch_size
		self.output_size=output_size
		self.hidden_size=hidden_size
		self.vocab_size=len(model_prim.wv.vocab)
		self.embedding_length=embedding_length

		self.word_embeddings=nn.Embedding.from_pretrained(torch.FloatTensor(model_prim.wv.vectors))

		# self.word_embeddings=nn.Embedding(vocab_size, embedding_length)
		
		if(weights is None):
			weights=[]

		if(len(weights)!=0):
			self.word_embeddings.weight=nn.Parameter(weights, requires_grad=False)

		self.lstm=nn.LSTM(embedding_length, hidden_size)
		self.label=nn.Linear(hidden_size, output_size)

	def embedding_(self, input_sentence):
		return self.word_embeddings(input_sentence)
		
	def forward(self, input_sentence, batch_size=None):
		input_=self.embedding_(input_sentence)

		if batch_size is None:
			h_0=Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) 
			c_0=Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
		else:
			h_0=Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			c_0=Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		# if batch_size is None:
		# 	h_0=Variable(torch.zeros(1, self.batch_size, self.hidden_size)) 
		# 	c_0=Variable(torch.zeros(1, self.batch_size, self.hidden_size))
		# else:
		# 	h_0=Variable(torch.zeros(1, batch_size, self.hidden_size))
		# 	c_0=Variable(torch.zeros(1, batch_size, self.hidden_size))
		output, (final_hidden_state, final_cell_state)=self.lstm(input_, (h_0, c_0))

		# final_output=self.label(final_hidden_state[-1])		
		final_output=self.label(output[-1])		
		return final_output, final_hidden_state, final_cell_state


class AuxTaskNetwork(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights=None):
		super(AuxTaskNetwork, self).__init__()
		
		self.batch_size=batch_size
		self.output_size=output_size
		self.hidden_size=hidden_size
		self.vocab_size=len(model_aux.wv.vocab)
		self.embedding_length=embedding_length

		self.word_embeddings=nn.Embedding.from_pretrained(torch.FloatTensor(model_aux.wv.vectors))
		# self.word_embeddings=nn.Embedding(vocab_size, embedding_length)
		
		if(weights is None):
			weights=[]

		if(len(weights)!=0):
			self.word_embeddings.weight=nn.Parameter(weights, requires_grad=False)

		self.lstm=nn.LSTM(embedding_length, hidden_size)
		self.label=nn.Linear(hidden_size, output_size)

	def embedding_(self, input_sentence):
		return self.word_embeddings(input_sentence)
		
	def forward(self, input_sentence, batch_size=None):
		input_=self.embedding_(input_sentence)
		
		if batch_size is None:
			h_0=Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) 
			c_0=Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
		else:
			h_0=Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			c_0=Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		output, (final_hidden_state, final_cell_state)=self.lstm(input_, (h_0, c_0))

		# final_output=self.label(final_hidden_state[-1])
		final_output=self.label(output[-1])
		
		return final_output, final_hidden_state, final_cell_state

class AuxTransferNetwork(nn.Module):
	def __init__(self, batch_size, hidden_size, embedding_length):
		super(AuxTransferNetwork, self).__init__()
		self.batch_size=batch_size
		self.hidden_size=hidden_size
		self.seq_len=FLAGS.max_sen_len
		self.embedding_length=embedding_length

		self.lstm=nn.LSTM(embedding_length, hidden_size)
		
	def forward(self, composite_cell_state, composite_hidden_state, batch_size=None):
		input_=torch.zeros((FLAGS.max_sen_len, self.batch_size, self.embedding_length))

		if(torch.cuda.is_available()):
			input_=input_.cuda()

		output, (final_hidden_state, final_cell_state)=self.lstm(input_, (composite_hidden_state, composite_cell_state))		
		return final_hidden_state, final_cell_state

class WeightAlignmentLayer(nn.Module):
	def __init__(self, batch_size, num_aux, hidden_size, output_size):
		super(WeightAlignmentLayer, self).__init__()
		self.num_aux=num_aux
		self.batch_size=batch_size
		self.output_size=output_size

		self.sigmoid=nn.Sigmoid()
		self.fc_cell_alpha=nn.Linear(2*hidden_size, 1)
		self.fc_hidden_alpha=nn.Linear(2*hidden_size, 1)
		self.label=nn.Linear(hidden_size, output_size)

	def forward(self, aux_cell_state, aux_hidden_state, composite_cell_state, composite_hidden_state, prim_hidden_state):
		if(self.num_aux==1 or not isinstance(aux_cell_state, list)):
			# aux_cell_state is one tensor, aux_hidden_state is one tensor
			inp_alpha_c=torch.cat((prim_hidden_state, aux_cell_state), -1)
			inp_alpha_h=torch.cat((prim_hidden_state, aux_hidden_state), -1)

			inp_alpha_c=inp_alpha_c.view((FLAGS.batch_size, inp_alpha_c.shape[-1]))
			inp_alpha_h=inp_alpha_h.view((FLAGS.batch_size, inp_alpha_h.shape[-1]))

			if torch.cuda.is_available():
				inp_alpha_c=inp_alpha_c.cuda()
				inp_alpha_h=inp_alpha_h.cuda()

			alpha_c=self.sigmoid(self.fc_cell_alpha(inp_alpha_c))
			alpha_h=self.sigmoid(self.fc_hidden_alpha(inp_alpha_h))

			reshaped_aux_cell_state=aux_cell_state.view((FLAGS.batch_size, aux_cell_state.shape[-1]))
			reshaped_aux_hidden_state=aux_hidden_state.view((FLAGS.batch_size, aux_hidden_state.shape[-1]))

			a_c=alpha_c*reshaped_aux_cell_state
			a_h=alpha_h*reshaped_aux_hidden_state

			beta_h=a_h/alpha_h
			beta_c=a_c/alpha_c

			new_composite_cell_state=a_c+composite_cell_state.view((FLAGS.batch_size, composite_cell_state.shape[-1]))*beta_c
			new_composite_hidden_state=a_h+composite_hidden_state.view((FLAGS.batch_size, composite_hidden_state.shape[-1]))*beta_h

			new_composite_cell_state, new_composite_hidden_state=new_composite_cell_state.view((1, FLAGS.batch_size, composite_cell_state.shape[-1])), new_composite_hidden_state.view((1, FLAGS.batch_size, composite_hidden_state.shape[-1]))

			final_output=self.label(new_composite_hidden_state[-1])
			return final_output, new_composite_cell_state, new_composite_hidden_state
		else:
			# aux_cell_state is an array of tensors, aux_hidden_state is an array of tensors
			inp_alpha_h=[]
			inp_alpha_c=[]

			for i in range(len(aux_cell_state)):
				inp_alpha_c.append(torch.cat((prim_hidden_state, aux_cell_state[i]), -1))
				inp_alpha_h.append(torch.cat((prim_hidden_state, aux_hidden_state[i]), -1))

			for i in range(len(aux_cell_state)):
				inp_alpha_c[i]=inp_alpha_c[i].view((FLAGS.batch_size, inp_alpha_c[i].shape[-1]))
				inp_alpha_h[i]=inp_alpha_h[i].view((FLAGS.batch_size, inp_alpha_h[i].shape[-1]))

			if torch.cuda.is_available():
				for i in range(len(aux_cell_state)):
					inp_alpha_c[i]=inp_alpha_c[i].cuda()
					inp_alpha_h[i]=inp_alpha_h[i].cuda()

			alpha_c=[]
			alpha_h=[]

			for i in range(len(aux_cell_state)):
				alpha_c.append(self.sigmoid(self.fc_cell_alpha(inp_alpha_c[i])))
				alpha_h.append(self.sigmoid(self.fc_hidden_alpha(inp_alpha_h[i])))

			reshaped_aux_cell_state=[]
			reshaped_aux_hidden_state=[]

			for i in range(len(aux_cell_state)):
				reshaped_aux_cell_state.append(aux_cell_state[i].view((FLAGS.batch_size, aux_cell_state[i].shape[-1])))
				reshaped_aux_hidden_state.append(aux_hidden_state[i].view((FLAGS.batch_size, aux_hidden_state[i].shape[-1])))

			a_c=[]
			a_h=[]

			for i in range(len(aux_cell_state)):
				a_c.append(alpha_c[i]*reshaped_aux_cell_state[i])
				a_h.append(alpha_h[i]*reshaped_aux_hidden_state[i])

			num_c=a_c[0]
			denom_c=alpha_c[0]

			num_h=a_h[0]
			denom_h=alpha_h[0]
			
			for i in range(1, len(aux_cell_state)):
				num_c=num_c+a_c[i]
				num_h=num_h+a_h[i]
				denom_c=denom_c+alpha_c[i]
				denom_h=denom_h+alpha_h[i]

			beta_h=num_h/denom_h
			beta_c=num_c/denom_c

			new_composite_cell_state=num_c+composite_cell_state.view((FLAGS.batch_size, composite_cell_state.shape[-1]))*beta_c
			new_composite_hidden_state=num_h+composite_hidden_state.view((FLAGS.batch_size, composite_hidden_state.shape[-1]))*beta_h

			new_composite_cell_state, new_composite_hidden_state=new_composite_cell_state.view((1, FLAGS.batch_size, composite_cell_state.shape[-1])), new_composite_hidden_state.view((1, FLAGS.batch_size, composite_hidden_state.shape[-1]))

			final_output=self.label(new_composite_hidden_state[-1])
			return final_output, new_composite_cell_state, new_composite_hidden_state