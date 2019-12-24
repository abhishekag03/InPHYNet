# InPHYNet
This repository contains the code and dataset for the InPHYNet network. The code was tested on Python 3.6.8, PyTorch 1.3.1 and Keras 2.3.1 with Tensorflow 1.14.0 backend.

## Getting Started
Clone this repository using the following command:
```
git clone https://github.com/abhishekag03/InPHYNet/
```
- The Physics data and TREC data can be found at this [link](https://drive.google.com/drive/folders/1nT0NlNJXljP74h4O7-3FjYLR83G1SdPt?usp=sharing). Download this data and put it within `data` folder. Your final data folder hierarchy should contain `<root-folder>/data/paragraphs.pickle`, `<root-folder>/data/labels.pickle` and `<root-folder>/data/aux_test_data_with_labels.txt`.
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
- TFIDF

To train and evaluate the baseline models on Doc2Vec embeddings, use the following command:
```
python baseline_doc2vec.py
```
To train and evaluate the baseline models on TFIDF embeddings, use the following command:
```
python baseline_tfidf.py
```
The results of baseline models for both of Doc2Vec and TFIDF embeddings are present in baseline_doc2vec_results.txt and baseline_tfidf_results.txt respectively.

## Training
To train the vanilla LSTM on Doc2Vec embeddings, use the following command:
```
python train_vanilla_lstm_doc2vec.py
```
To train the vanilla LSTM on TFIDF embeddings, use the following command:
```
python train_vanilla_lstm_tfidf.py
```
To train and evaluate InPHYNet with a single aux task (TREC dataset) on Doc2Vec embeddings, use the following command:
```
python train_inPHYNet_doc2vec.py
```
The evaluation results will be printed after every epoch on the heldout test set.
```
To train and evaluate InPHYNet with a single aux task (TREC dataset) on TFIDF embeddings, use the following command:
```
python train_inPHYNet_tfidf.py
```
The evaluation results will be printed after every epoch on the heldout test set.

To train GIRNet on Doc2Vec embeddings, use the following command:
```
python train_GIRNet_doc2vec.py
```
To train GIRNet on TFIDF embeddings, use the following command:
```
python train_GIRNet_tfidf.py
```

## Evaluation
To test the vanilla LSTM performance on Doc2Vec embeddings, use the following command:
```
python evaluate_vanilla_lstm_doc2vec.py
```
To test the vanilla LSTM performance on TFIDF embeddings, use the following command:
```
python evaluate_vanilla_lstm_tfidf.py
```
To test InPHYNet performance on a singe aux task (TREC dataset) on Doc2Vec embeddings, use the following command:
```
python train_inPHYNet_doc2vec.py
```
The evaluation results will be printed after every epoch on the heldout test set.
To test InPHYNet performance on a singe aux task (TREC dataset) on TFIDF embeddings, use the following command:
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

In case of questions, contact: 
- vishaal16119 [at] iiitd [dot] ac [dot] in
- abhishek16126 [at] iiitd [dot] ac [dot] in
- mohitr [at] iiitd [dot] ac [dot] in
