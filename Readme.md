# InPHYNet
This repository contains the code and dataset for the paper titled ["InPHYNet: Leveraging attention-based multitask recurrent networks for multi-label physics text classification"](https://www.sciencedirect.com/science/article/pii/S095070512030616X). The code was tested on Python 3.6.8, PyTorch 1.3.1 and Keras 2.3.1 with Tensorflow 1.14.0 backend.

## Getting Started
Clone this repository using the following command:
```
git clone https://github.com/abhishekag03/InPHYNet/
```
- The TREC data can be found at this [link](https://cogcomp.seas.upenn.edu/Data/QA/QC/). Download this data and put it within `data` folder.
- The SST data is directly loaded from the official torchtext repository. For official documentation, refer [this](https://torchtext.readthedocs.io/en/latest/datasets.html#sst).
- Install all the dependencies using the following command:
```
pip install -r requirements.txt
```
- Download the pretrained Spacy English model using the following command:
```
python -m spacy download en_core_web_sm
```
- Install the `stopwords` in the `nltk` installation using the following commands within your python interpretor:
```
>>> import nltk
>>> nltk.download('stopwords')
```

## Baselines
We use the following models as our baseline models for the primary Physics text multi-label classification:
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- Multi Layer Perceptron
- Multi-Label KNN

We also utilise the following label transformation techniques to convert the multi-label problem into a more feasible multi-class one:
- Label Powerset
- Classifier Chain
- Binary Relevance

We use two types of input embeddings:
- Doc2Vec
- TF-IDF

To train and evaluate the baseline models on Doc2Vec embeddings, use the following command:
```
python baseline_doc2vec.py
```
To train and evaluate the baseline models on TF-IDF embeddings, use the following command:
```
python baseline_tfidf.py
```
The results of baseline models for both of Doc2Vec and TF-IDF embeddings are present in baseline_doc2vec_results.txt and baseline_tfidf_results.txt respectively.

## Training
To train the vanilla LSTM on Doc2Vec embeddings, use the following command:
```
python train_vanilla_lstm_doc2vec.py
```
To train the vanilla LSTM on TF-IDF embeddings, use the following command:
```
python train_vanilla_lstm_tfidf.py
```
To train and evaluate InPHYNet with a single auxiliary task (TREC dataset) on Doc2Vec embeddings, use the following command:
```
python train_inPHYNet_doc2vec.py
```
The evaluation results will be printed after every epoch on the heldout test set.

To train and evaluate InPHYNet with a single auxiliary task (TREC dataset) on TF-IDF embeddings, use the following command:
```
python train_inPHYNet_tfidf.py
```
The evaluation results will be printed after every epoch on the heldout test set.

To train GIRNet on Doc2Vec embeddings, use the following command:
```
python train_GIRNet_doc2vec.py
```
To train GIRNet on TF-IDF embeddings, use the following command:
```
python train_GIRNet_tfidf.py
```
For extracting features from the BERT baseline, you can use the following script:
```
from transformers import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
```
Once, the features are extracted, you can train an MLP network with BCE loss for the downstream primary task.

## Evaluation
To test the vanilla LSTM performance on Doc2Vec embeddings, use the following command:
```
python evaluate_vanilla_lstm_doc2vec.py
```
To test the vanilla LSTM performance on TF-IDF embeddings, use the following command:
```
python evaluate_vanilla_lstm_tfidf.py
```
To test InPHYNet performance on a singe aux task (TREC dataset) on Doc2Vec embeddings, use the following command:
```
python train_inPHYNet_doc2vec.py
```
The evaluation results will be printed after every epoch on the heldout test set.
To test InPHYNet performance on a singe aux task (TREC dataset) on TF-IDF embeddings, use the following command:
```
python train_inPHYNet_tfidf.py
```
The evaluation results will be printed after every epoch on the heldout test set.
To test GIRNet performance on Doc2Vec embeddings, use the following command:
```
python evaluate_GIRNet_doc2vec.py
```
To test GIRNet performance on TFIDF embeddings, use the following command:
```
python evaluate_GIRNet_tfidf.py
```
## Notes
- To change the hyperparameters and other tunable parameters, update the `flags.py` with appropriate changes
- Currently, the models are saved in the `checkpoints_<x>` folders and the result logs are updated in the `results_<x>` folders (where x indicates the trained model like baseline, vanilla_lstm, girnet, inphynet etc).
- The `num_aux` parameter in `flags.py` is used to indicate the number of auxiliary tasks that are used to train InPHYNet. Change this parameter accordingly when training for single or multiple tasks.

## Citation
If you find this work useful, please cite it as:
```
@article{udandarao2020inphynet,
  title={Inphynet: Leveraging attention-based multitask recurrent networks for multi-label physics text classification},
  author={Udandarao, Vishaal and Agarwal, Abhishek and Gupta, Anubha and Chakraborty, Tanmoy},
  journal={Knowledge-Based Systems},
  pages={106487},
  year={2020},
  publisher={Elsevier}
}
```

In case of questions, contact: 
- vishaal16119 [at] iiitd [dot] ac [dot] in
- abhishek16126 [at] iiitd [dot] ac [dot] in
