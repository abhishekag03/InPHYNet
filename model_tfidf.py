import os
import pickle
from dataloader_keras import *
from emb import *
from sklearn.feature_extraction.text import TfidfVectorizer

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
# tagged_data=[TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
tfidf_vec = TfidfVectorizer()
tfidf_count_occurs = tfidf_vec.fit_transform(data)
print(tfidf_vec.get_feature_names())
tfidf_vec.save("aux_tfidf.model")

#PRIM
tfidf_vec_prim = TfidfVectorizer()
tfidf_count_occurs = tfidf_vec_prim.fit_transform(paras_x)
print(tfidf_vec_prim.get_feature_names())