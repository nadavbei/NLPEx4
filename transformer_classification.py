import numpy as np
from matplotlib import pyplot as plt

from data_loader import SentimentTreeBank

category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # Initialize dataset
    dataset = SentimentTreeBank()

    # Get training, validation, and test sets
    train_set = dataset.get_train_set()
    validation_set = dataset.get_validation_set()
    test_set = dataset.get_test_set()

    # Get data
    train_data = [sent.text for sent in train_set]
    validation_data = [sent.text for sent in validation_set]
    test_data = [sent.text for sent in test_set]

    train_label = [sent.sentiment_val for sent in train_set]
    validation_label = [sent.sentiment_val for sent in validation_set]
    test_label = [sent.sentiment_val for sent in test_set]

    # Lengths
    train_len = int(portion * len(train_set))
    test_len = int(portion * len(test_set))

    # Train
    x_train = np.array(train_data[:train_len])
    y_train = np.array(train_label[:train_len])

    # Remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # Test
    x_test = np.array(test_data[:test_len])
    y_test = np.array(test_label[:test_len])

    # Remove empty entries
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()

    return x_train, y_train, x_test, y_test


def transformer_classification(portion=1.):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm import tqdm

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset for loading data
        """

        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        """
        Perform an epoch of training of the model with the optimizer
        :param model:
        :param data_loader:
        :param optimizer:
        :param dev:
        :return: Average loss over the epoch
        """
        model.train()
        total_loss = 0.
        # iterate over batches
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate_model(model, data_loader, dev='cpu', metric=None):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].to(dev)
                attention_mask = batch['attention_mask'].to(dev)
                labels = batch['labels'].to(dev)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        return correct / total

    x_train, y_train, x_test, y_test = get_data(portion=portion)

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = len(category_dict)
    epochs = 2
    batch_size = 64
    learning_rate = 1e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric = evaluate.load("accuracy")

    # Datasets and DataLoaders
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training and evaluation
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, dev)
        val_accuracy = evaluate_model(model, val_loader, dev)

        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.show()

    return model

if __name__ == '__main__':
    model = transformer_classification()
    print("success")