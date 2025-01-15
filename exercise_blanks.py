import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
from torch.utils.data import DataLoader, Dataset

import data_loader
import pickle
import tqdm
from data_loader import *

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
NEGATED = "negated"
RARE = "rare"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    # Access the text of the sentence
    words = sent.text

    # Initialize an array to accumulate embeddings
    embedding_sum = np.zeros(embedding_dim, dtype=np.float32)
    count = 0  # Counter for valid words with embeddings

    for word in words:
        if word in word_to_vec:  # Check if the word has a pre-trained embedding
            embedding_sum += word_to_vec[word]
            count += 1

    # Avoid division by zero
    if count == 0:
        return np.zeros(embedding_dim, dtype=np.float32)

    # Compute the average embedding
    return embedding_sum / count


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size, dtype=np.float32)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    sentence_tokens = sent.text
    vector_size = len(word_to_ind)
    average_embedding = np.zeros(vector_size, dtype=np.float32)
    valid_tokens_count = 0

    for token in sentence_tokens:
        if token in word_to_ind:
            index = word_to_ind[token]
            one_hot_vector = get_one_hot(vector_size, index)
            average_embedding += one_hot_vector
            valid_tokens_count += 1

    if valid_tokens_count > 0:
        average_embedding /= float(valid_tokens_count)

    return average_embedding


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: idx for idx, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    # Initialize the result array with zero vectors
    embeddings = np.zeros((seq_len, embedding_dim))

    # Map each word in the sentence to its embedding
    for i, word in enumerate(sent.text[:seq_len]):  # Consider only the first `seq_len` words
        embeddings[i] = word_to_vec.get(word, np.zeros(embedding_dim))  # Map to embedding or zero vector

    return embeddings


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        negated_sentences_indices = get_negated_polarity_examples(self.sentiment_dataset.get_test_set())
        negated_sentences = []
        for ind in negated_sentences_indices:
            negated_sentences.append(self.sentences[TEST][ind])
        self.sentences[NEGATED] = negated_sentences

        rare_sentences_indices = get_rare_words_examples(self.sentiment_dataset.get_test_set(), self.sentiment_dataset)
        rare_sentences = []
        for ind in rare_sentences_indices:
            rare_sentences.append(self.sentences[TEST][ind])
        self.sentences[RARE] = rare_sentences

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    class LSTM(nn.Module):

        """
        An LSTM for sentiment analysis with a bidirectional LSTM architecture.
        """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        """
        Initialize the LSTM model.
        :param embedding_dim: Dimension of the word embeddings
        :param hidden_dim: Dimension of the LSTM's hidden state
        :param n_layers: Number of LSTM layers
        :param dropout: Dropout rate
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define the bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,  # Dropout only if n_layers > 1
        )

        # Define a linear layer to map concatenated hidden states to a single value
        self.fc = nn.Linear(hidden_dim * 2, 1)

        # Define the sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        """
        Forward pass of the LSTM model.
        :param text: Input tensor of shape (batch_size, seq_len, embedding_dim)
        :return: Output tensor with probabilities (batch_size, 1)
        """
        # Pass the input through the LSTM layer
        text = text.float()
        lstm_out, (h_n, c_n) = self.lstm(text)

        # Concatenate the last hidden states from both directions
        last_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)

        # Pass through the fully connected layer
        linear_out = self.fc(last_hidden_state)

        # Apply the sigmoid activation
        output = self.sigmoid(linear_out)

        return output

    def predict(self, text):
        """
        Make a prediction for the input text.
        :param text: Input tensor of shape (batch_size, seq_len, embedding_dim)
        :return: Predicted probabilities (batch_size, 1)
        """
        with torch.no_grad():
            return self.forward(text)


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        logits = self.fc(x)
        return logits

    def predict(self, x):
        logits = self.forward(x)  # Get logits
        probs = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        predictions = (probs >= 0.5).long()  # Convert to binary predictions using threshold 0.5
        return predictions.squeeze()


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """

    # Round predictions to 0 or 1
    rounded_preds = np.round(preds)

    # Calculate the number of correct predictions
    correct = (rounded_preds == y).sum()

    # Calculate accuracy
    accuracy = correct / len(y)

    return accuracy


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """

    model.train()  # Set the model to training mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in data_iterator:
        inputs, targets = batch  # Unpack inputs and targets from the batch

        optimizer.zero_grad()  # Clear previous gradients

        outputs = model(inputs)  # Forward pass

        loss = criterion(outputs.squeeze(), targets.float())  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        total_loss += loss.item()  # Accumulate the loss

        # Compute predictions and accuracy
        predicted = (torch.sigmoid(outputs) >= 0.5).long()  # Apply sigmoid and threshold
        correct_predictions += (predicted.squeeze() == targets).sum().item()
        total_samples += targets.size(0)

    # Calculate average loss and accuracy
    average_loss = total_loss / len(data_iterator)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch in data_iterator:
            inputs, targets = batch  # Unpack inputs and targets from the batch

            outputs = model(inputs)  # Forward pass
            # print("outputs:", outputs.shape)
            # print("targets:", targets.float().shape)
            loss = criterion(outputs.squeeze(), targets.float())  # Compute the loss
            total_loss += loss.item()  # Accumulate the loss

            # Compute predictions and accuracy (apply sigmoid + threshold)
            predicted = (torch.sigmoid(outputs) >= 0.5).long()
            correct_predictions += (predicted.squeeze() == targets).sum().item()
            total_samples += targets.size(0)

    # Calculate average loss and accuracy
    average_loss = total_loss / len(data_iterator)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def evaluate_test(model, data_manager, data_subset=TEST):
    """
    Evaluates the model on the test dataset.
    :param model: the trained model.
    :param data_manager: the DataManager object containing the test data.
    :param criterion: the loss function (e.g., CrossEntropyLoss).
    :return: test_loss, test_accuracy
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loader = data_manager.get_torch_iterator(data_subset=data_subset)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    return test_loss, test_accuracy


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device of the model
    predictions = []

    with torch.no_grad():  # No need to compute gradients for inference
        for batch in data_iter:
            inputs, _ = batch  # We only need the inputs for predictions
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            predicted = model.predict(inputs)  # Use the model's predict method
            predictions.append(predicted.cpu())  # Collect predictions on CPU

    return torch.cat(predictions).numpy()


def train_model(model, data_manager: DataManager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    # Initialize the optimizer with Adam, using the model's parameters, learning rate, and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Use cross-entropy loss for classification tasks
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()

    # Get the training and validation data loaders from DataManager
    train_loader = data_manager.get_torch_iterator(data_subset=TRAIN)
    val_loader = data_manager.get_torch_iterator(data_subset=VAL)

    # Track training and validation metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")

        # Training phase
        model.train()  # Set the model to training mode
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Log epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}")

    # Save the results in a dictionary for later analysis or plotting
    training_results = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }

    return training_results


def train_log_linear_with_one_hot(model, data_manager, weight_decay, n_epochs=20, lr=0.01):
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    # Train the model using the previously defined `train_model` function
    training_results = train_model(model, data_manager, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay)
    return training_results


def train_log_linear_with_w2v(model, data_manager, weight_decay, n_epochs=20, lr=0.01):
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    training_results = train_model(model, data_manager, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay)
    return training_results


def train_lstm_with_w2v(model, data_manager, weight_decay, n_epochs=20, lr=0.01):
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    training_results = train_model(model, data_manager, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay)
    return training_results


def plot_losses(training_results):
    """
    Plots the training and validation loss curves.
    :param training_results: Dictionary containing 'train_losses' and 'val_losses'.
    """
    epochs = range(1, len(training_results['train_losses']) + 1)

    plt.figure(figsize=(10, 6))

    # Plot train and validation losses
    plt.plot(epochs, training_results['train_losses'], label='Train Loss', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, training_results['val_losses'], label='Validation Loss', color='red', linestyle='--', marker='x')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epochs')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

def plot_accuracies(training_results):
    """
    Plots the training and validation accuracy curves.
    :param training_results: Dictionary containing 'train_accuracies' and 'val_accuracies'.
    """
    epochs = range(1, len(training_results['train_accuracies']) + 1)

    plt.figure(figsize=(10, 6))

    # Plot train and validation accuracies
    plt.plot(epochs, training_results['train_accuracies'], label='Train Accuracy', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, training_results['val_accuracies'], label='Validation Accuracy', color='red', linestyle='--', marker='x')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs Epochs')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


def log_linear_model(type):
    data_manager = None
    if type == ONEHOT_AVERAGE:
        data_manager = DataManager(ONEHOT_AVERAGE, batch_size=64)
    elif type == W2V_AVERAGE:
        data_manager = DataManager(W2V_AVERAGE, batch_size=64, embedding_dim=300)
    input_dim = data_manager.get_input_shape()[0]  # Assuming one-hot encoding, it is the length of the sentence vector
    model = LogLinear(embedding_dim=input_dim)
    training_results = train_log_linear_with_one_hot(model, data_manager, weight_decay=0.001)

    # a,b:
    plot_losses(training_results)
    plot_accuracies(training_results)

    # a:
    test_loss, test_accuracy = evaluate_test(model, data_manager)
    print("test_loss =", test_loss)
    print("test_accuracy =", test_accuracy)

    # b:
    negated_loss, negated_accuracy = evaluate_test(model, data_manager, NEGATED)
    print("negated_loss =", negated_loss)
    print("negated_accuracy =", negated_accuracy)

    rare_loss, rare_accuracy = evaluate_test(model, data_manager, RARE)
    print("rare_loss =", rare_loss)
    print("rare_accuracy =", rare_accuracy)


def lstm_model():
    data_manager = DataManager(W2V_SEQUENCE, batch_size=64, embedding_dim=300)
    model = LSTM(embedding_dim=300, hidden_dim=100, n_layers=52, dropout=0.5)

    training_results = train_lstm_with_w2v(model, data_manager, weight_decay=0.0001, n_epochs=4, lr=0.001)

    # a,b:
    plot_losses(training_results)
    plot_accuracies(training_results)

    # a:
    test_loss, test_accuracy = evaluate_test(model, data_manager)
    print("test_loss =", test_loss)
    print("test_accuracy =", test_accuracy)

    # b:
    negated_loss, negated_accuracy = evaluate_test(model, data_manager, NEGATED)
    print("negated_loss =", negated_loss)
    print("negated_accuracy =", negated_accuracy)

    rare_loss, rare_accuracy = evaluate_test(model, data_manager, RARE)
    print("rare_loss =", rare_loss)
    print("rare_accuracy =", rare_accuracy)

if __name__ == '__main__':
    # log_linear_model(ONEHOT_AVERAGE)
    log_linear_model(W2V_AVERAGE)
    # lstm_model()
