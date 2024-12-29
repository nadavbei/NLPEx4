import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Training Function
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

# Evaluation Function
def evaluate_model(model, data_loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return correct / total

# Main Function
def train_transformer(x_train, y_train, x_test, y_test, learning_rate, weight_decay, batch_size, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, tokenizer, and optimizer
    model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Tokenize the data
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=512)

    # Create DataLoaders
    train_dataset = SentimentDataset(train_encodings, y_train)
    test_dataset = SentimentDataset(test_encodings, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_accuracy = evaluate_model(model, test_loader, device)

        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Plot Training Results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

if __name__ == "__main__":
    # Example Data (Replace with actual dataset loading)
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split

    # Load data
    categories = ["sci.space", "rec.sport.hockey"]
    data = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))
    x_data, y_data = data.data, data.target

    # Split data into train/test sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Hyperparameters
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0
    BATCH_SIZE = 50
    EPOCHS = 2

    # Train Transformer
    trained_model = train_transformer(x_train, y_train, x_test, y_test, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, EPOCHS)
