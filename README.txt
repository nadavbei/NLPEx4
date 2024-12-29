Overview
This project implements and compares three different models for sentiment analysis using PyTorch:

Simple Log-Linear Model – A basic model using one-hot embeddings.
Word2Vec Log-Linear Model – A model using pre-trained Word2Vec embeddings.
LSTM Model – A sophisticated model utilizing Bi-directional LSTM layers with Word2Vec embeddings.
Dataset
The dataset used for this sentiment analysis task is the Sentiment Treebank dataset by Stanford.
It includes sentences from movie reviews with sentiment values (ranging from 0 to 1).
Sentences with sentiment values between 0.4 and 0.6 are discarded, leaving only binary classifications (0 for negative, 1 for positive).

Special Subsets:
Negated Polarity – Sentences where the polarity of a sub-phrase negates the overall sentiment.
Rare Words – Sentences with the lowest frequency of rare words with non-neutral sentiment.
Data Loading
The data_loader.py script provides functionalities for loading the dataset, preprocessing sentences,
and splitting the data into train, validation, and test sets. This includes handling sentences,
 subs-sentences, and their corresponding binary labels.

Models
1. Simple Log-Linear Model
Implementation: LogLinear class
Process: Average one-hot embeddings → Linear Layer → Sigmoid
2. Word2Vec Log-Linear Model
Implementation: LogLinear class (same as Simple Log-Linear, but uses pre-trained Word2Vec embeddings)
Process: Average Word2Vec embeddings → Linear Layer → Sigmoid
3. LSTM Model
Architecture: Bi-directional LSTM → Linear Layer → Sigmoid
Embedding: Word2Vec embeddings