import pandas as pd
# %%
import pandas as pd
import numpy as np
import re

# Load the data
train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")

import string


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.split()


train['content'] = train['content'].apply(preprocess_text)
val['content'] = val['content'].apply(preprocess_text)
test['content'] = test['content'].apply(preprocess_text)
# %%
train
# %%
print(f"Shape of the train data: {train.shape}")
print(f"Shape of the validation data: {val.shape}")
print(f"Shape of the test data: {test.shape}")
# %%
# import numpy as np



class NgramLanguageModel:
    def __init__(self, n):
        self.n = n
        self.counts = {}
        self.counts_minus_one_grams = {}
        self.vocab = set()

    def update_counts(self, tokens):
        n = self.n
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            self.counts[ngram] = self.counts.get(ngram, 0) + 1
            for token in ngram:
                self.vocab.add(token)

        n = self.n - 1
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            self.counts_minus_one_grams[ngram] = self.counts_minus_one_grams.get(ngram, 0) + 1

    def probability(self, token, context):
        context = tuple(context)
        ngram = context + (token,)
        if context in self.counts_minus_one_grams:
            context_count = self.counts_minus_one_grams[context]
            if ngram in self.counts:
                return self.counts[ngram] / context_count
        return 0

    def perplexity(self, test_data):
        cross_entropy = 0
        total_ngrams = 0
        for i in range(len(test_data) - self.n + 1):
            context = tuple(test_data[i:i + self.n - 1])
            token = test_data[i + self.n - 1]
            prob = self.probability(token, context)
            if prob > 0:  # Avoid log(0)
                cross_entropy += prob * np.log2(prob)
                total_ngrams += 1
        perplexity = 2 ** (- cross_entropy)
        return perplexity


# Train the model
ngram_model = NgramLanguageModel(2)  # Change the number to the desired n-gram
for tokens in train['content']:
    ngram_model.update_counts(tokens)

print("I have finished counting the n-grams")
# Calculate probabilities
for tokens in train['content']:
    for i in range(len(tokens) - ngram_model.n + 1):
        context = tuple(tokens[i:i + ngram_model.n - 1])
        token = tokens[i + ngram_model.n - 1]
        a = ngram_model.probability(token, context)

# Evaluate the model
test_tokens = [token for sublist in test['content'].tolist() for token in sublist]  # Flatten the list of tokens
print(f"Perplexity of the model on the test data: {ngram_model.perplexity(test_tokens)}")
