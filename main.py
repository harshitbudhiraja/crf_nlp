import pandas as pd
import spacy
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

nltk.download('punkt')  

nlp = spacy.load('en_core_web_sm')

class SimpleCRF(nn.Module):
    def __init__(self, num_tags):
        super(SimpleCRF, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask=None):
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return -log_likelihood

    def _compute_log_likelihood(self, emissions, tags, mask):
        seq_length, batch_size, num_tags = emissions.size()
        score = self.start_transitions[tags[0]] + emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            transition_score = self.transitions[tags[i - 1], tags[i]]
            emission_score = emissions[i, torch.arange(batch_size), tags[i]]
            score += (transition_score + emission_score) * mask[i]

        last_tag_indices = mask.sum(dim=0) - 1
        last_tags = tags.gather(0, last_tag_indices.unsqueeze(0)).squeeze(0)
        score += self.end_transitions[last_tags]

        return score.sum()

    def _viterbi_decode(self, emissions, mask):
        seq_length, batch_size, num_tags = emissions.size()
        viterbi_score = self.start_transitions + emissions[0]
        viterbi_path = torch.zeros_like(emissions, dtype=torch.long)

        for i in range(1, seq_length):
            broadcast_score = viterbi_score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            score = broadcast_score + self.transitions + broadcast_emission
            best_score, best_path = score.max(dim=1)
            viterbi_score = best_score * mask[i].unsqueeze(1) + viterbi_score * (1 - mask[i].unsqueeze(1))
            viterbi_path[i] = best_path

        last_tag_indices = mask.sum(dim=0) - 1
        best_tags = [viterbi_path[last_tag_indices[i], i].item() for i in range(batch_size)]

        return best_tags

    def fit(self, emissions, tags, mask=None, epochs=10, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            loss = self.forward(emissions, tags, mask)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def predict(self, emissions, mask=None):
        self.eval()
        with torch.no_grad():
            return self._viterbi_decode(emissions, mask)


def load_data(train_path, test_path, use_cache=True, cache_dir='cache'):
    """Load and parse the NER dataset using spaCy with caching support."""

    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, 'processed_data.pkl')
    
    if use_cache and os.path.exists(cache_file):
        print(f"Loading data from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}. Processing data from scratch.")
    
    print("Processing data from source files...")
    start_time = time.time()
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    def parse_dataframe(df):

        docs = list(nlp.pipe(df['Sentence'].tolist()))
        sentences = [[token.text for token in doc] for doc in docs]
        pos_tags = [[token.pos_ for token in doc] for doc in docs]
        ner_tags = df['Tag'].apply(eval).tolist()
        return sentences, pos_tags, ner_tags
    
    result = (parse_dataframe(train_data), parse_dataframe(test_data))
    
    if use_cache:
        print(f"Saving processed data to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    
    print(f"Data processing completed in {time.time() - start_time:.2f} seconds")
    return result

def word2features(sent, pos, i):
    """Extract features for a given word."""
    word = sent[i]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'pos': pos[i],
        'word.shape': nlp(word)[0].shape_, 
        'word.prefix': word[:3], 
        'word.suffix': word[-3:] if len(word) > 3 else word, 
    }
    
    if i > 0:
        prev_word = sent[i-1]
        prev_pos = pos[i-1]
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:pos': prev_pos,
        })
    else:
        features['BOS'] = True
    
    # Next word features
    if i < len(sent) - 1:
        next_word = sent[i+1]
        next_pos = pos[i+1]
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:pos': next_pos,
        })
    else:
        features['EOS'] = True
    
    return features

def prepare_data(sentences, pos_tags, ner_tags, use_cache=True, cache_name='features', cache_dir='cache'):
    """Convert raw data into CRF features format with caching support."""

    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f'{cache_name}.pkl')
    
    if use_cache and os.path.exists(cache_file):
        print(f"Loading features from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}. Preparing features from scratch.")
    
    print(f"Preparing {cache_name} features...")
    start_time = time.time()
    
    X = []
    y = []
    
    for sent, pos, tags in zip(sentences, pos_tags, ner_tags):

        if isinstance(sent, str):
            sent = sent.split()
            
        sent_features = [word2features(sent, pos, i) for i in range(len(sent))]
        
        X.append(sent_features)
        y.append(tags)
    
    result = (X, y)
    
    if use_cache:
        print(f"Saving features to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    
    print(f"Feature preparation completed in {time.time() - start_time:.2f} seconds")
    return result

def train_evaluate_crf(X_train, Y_train, X_test, Y_test):
    """Train CRF model and evaluate its performance."""

    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    print("Training CRF model...")
    crf.fit(X_train, Y_train)
    
    Y_pred = crf.predict(X_test)
    
    metrics_dict = {
        'precision': metrics.flat_precision_score(Y_test, Y_pred, average='weighted'),
        'recall': metrics.flat_recall_score(Y_test, Y_pred, average='weighted'),
        'f1': metrics.flat_f1_score(Y_test, Y_pred, average='weighted')
    }
    
    return crf, Y_pred, metrics_dict

def main():

    print("Loading data...")
    (train_sentences, train_pos, train_tags), (test_sentences, test_pos, test_tags) = load_data(
        'data/ner_train.csv', 
        'data/ner_test.csv',
        use_cache=True
    )

    print("Preparing features...")
    X_train, Y_train = prepare_data(train_sentences, train_pos, train_tags, use_cache=True, cache_name='train_features')
    X_test, Y_test = prepare_data(test_sentences, test_pos, test_tags, use_cache=True, cache_name='test_features')

    # Hacky way to match lengths
    for i in range(len(X_train)):
        min_len = min(len(X_train[i]), len(Y_train[i]))
        X_train[i] = X_train[i][:min_len]  
        Y_train[i] = Y_train[i][:min_len]  

    for i in range(len(X_test)):
        min_len = min(len(X_test[i]), len(Y_test[i]))
        X_test[i] = X_test[i][:min_len]  
        Y_test[i] = Y_test[i][:min_len]  

    # Train and evaluate
    crf, predictions, metrics = train_evaluate_crf(X_train, Y_train, X_test, Y_test)
    
    # Print results
    print("\nModel Performance:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Print some example predictions
    # print("\nExample predictions:")
    # for i in range(min(3, len(test_sentences))):
    #     if isinstance(test_sentences[i], str):
    #         sent = test_sentences[i].split()
    #     else:
    #         sent = test_sentences[i]
    #     print(f"Sentence: {' '.join(sent)}")
    #     print(f"True tags: {test_tags[i]}")
    #     print(f"Predicted: {predictions[i]}")
    #     print()

if __name__ == "__main__":
    main()