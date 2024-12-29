from exercise_blanks import *
from data_loader import *

# Initialize dataset
dataset = SentimentTreeBank()

# Get training, validation, and test sets
train_set = dataset.get_train_set()
validation_set = dataset.get_validation_set()
test_set = dataset.get_test_set()

# Extract subsets
negated_indices = get_negated_polarity_examples(test_set)
negated_subset = [test_set[i] for i in negated_indices]

rare_indices = get_rare_words_examples(test_set, dataset)
rare_subset = [test_set[i] for i in rare_indices]

# Train models
models = {
    "log_linear_one_hot": train_log_linear_with_one_hot(),
    "log_linear_word2vec": train_log_linear_with_w2v(),
    "lstm": train_lstm_with_w2v(),
    "transformer": train_transformer(train_set, validation_set)
}

# Define loss criterion
loss = nn.CrossEntropyLoss()

# Get torch data iterators
train_iterator = DataLoader(train_set, batch_size=32, shuffle=True)
validation_iterator = DataLoader(validation_set, batch_size=32)
test_iterator = DataLoader(test_set, batch_size=32)

# Evaluate models
results = {}

for model_name, model in models.items():
    results[model_name] = {
        "test_accuracy": evaluate(model, test_iterator, loss),
        "validation_accuracy": evaluate(model, validation_iterator, loss),
        "negated_accuracy": evaluate(model, DataLoader(negated_subset, batch_size=32), loss),
        "rare_accuracy": evaluate(model, DataLoader(rare_subset, batch_size=32), loss)
    }

# Compare simple log-linear model vs Word2Vec log-linear model
simple_vs_word2vec = {
    "log_linear_one_hot": results['log_linear_one_hot'],
    "log_linear_word2vec": results['log_linear_word2vec']
}

best_simple_vs_word2vec = max(simple_vs_word2vec, key=lambda x: simple_vs_word2vec[x]['test_accuracy'])
print(f"Best performer (Simple vs Word2Vec): {best_simple_vs_word2vec}")

# Compare LSTM and Transformer models
lstm_vs_transformer = {
    "lstm": results['lstm'],
    "transformer": results['transformer']
}

best_lstm_vs_transformer = max(lstm_vs_transformer, key=lambda x: lstm_vs_transformer[x]['test_accuracy'])
print(f"Best performer (LSTM vs Transformer): {best_lstm_vs_transformer}")

# Special subsets comparison
special_subsets = {
    "negated": negated_subset,
    "rare": rare_subset
}

for subset_name, subset in special_subsets.items():
    best_model = max(models, key=lambda x: results[x][f'{subset_name}_accuracy'])
    worst_model = min(models, key=lambda x: results[x][f'{subset_name}_accuracy'])
    print(f"Best model on {subset_name} subset: {best_model}")
    print(f"Worst model on {subset_name} subset: {worst_model}")
