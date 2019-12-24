from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from nltk.tokenize import word_tokenize
import os
import pickle
from dataloader_keras import *
from emb import *

aux_data='data/aux_data_with_labels.txt'
PREPROCESSED_AUX_DATA={'x':'data/preprocessed_aux_paras.pickle', 'y':'data/preprocessed_aux_labels.pickle'}

MAX_EPOCHS=100
VEC_SIZE=300
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

print(paras_x[:5], paras_y[:5])
print(labels_aux[:5], paras_aux[:5])
#AUX

data=paras_aux
tagged_data=[TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

model_dm=Doc2Vec(vector_size=VEC_SIZE, alpha=ALPHA, min_alpha=0.00025, min_count=1, dm =1, window=10)
model_dm.build_vocab(tagged_data)

for epoch in range(MAX_EPOCHS):
    print('iteration {0} for Distributed Memory'.format(epoch))
    model_dm.train(tagged_data, total_examples=model_dm.corpus_count, epochs=model_dm.iter)
    model_dm.alpha -= 0.0002
    model_dm.min_alpha = model_dm.alpha

# model_dbow=Doc2Vec(vector_size=VEC_SIZE, alpha=ALPHA, min_alpha=0.00025, min_count=1, dm=0, window=10)
# model_dbow.build_vocab(tagged_data)

# for epoch in range(MAX_EPOCHS):
#     print('iteration {0} for Distributed bag of Words'.format(epoch))
#     model_dbow.train(tagged_data, total_examples=model_dbow.corpus_count, epochs=model_dbow.iter)
#     model_dbow.alpha -= 0.0002
#     model_dbow.min_alpha = model_dbow.alpha

# final_model=ConcatenatedDoc2Vec([model_dbow, model_dm])

model_dm.save("aux_dm_INPHYNET.model")
# model_dbow.save("data_d2v_aux_dbow.model")
print("Models Saved as paras_d2v_aux_dm.model and paras_d2v_aux_dbow.model")

# # test_data=word_tokenize("I love chatbots".lower())
# # v1=final_model.infer_vector(test_data)
# # print("V1_infer", len(v1))