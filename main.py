import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

class CRF:
    def __init__(self, num_features, num_labels):
        """
        Initialize the CRF model.
        
        Parameters:
        -----------
        num_features : int
            Number of features
        num_labels : int
            Number of possible labels
        """
        self.num_features = num_features
        self.num_labels = num_labels
        self.weights = np.random.randn(num_features * num_labels + num_labels * num_labels)
        
    def _get_feature_weights(self):
        """Extract feature weights from the weights vector"""
        return self.weights[:self.num_features * self.num_labels].reshape(self.num_features, self.num_labels)
    
    def _get_transition_weights(self):
        """Extract transition weights from the weights vector"""
        return self.weights[self.num_features * self.num_labels:].reshape(self.num_labels, self.num_labels)
    
    def compute_potential(self, x, y):
        """
        Compute potential for a single sequence.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Feature matrix of shape (sequence_length, num_features)
        y : numpy.ndarray
            Label sequence of shape (sequence_length,)
        
        Returns:
        --------
        float
            Potential score for this sequence
        """
        feature_weights = self._get_feature_weights()
        transition_weights = self._get_transition_weights()
        
        # Compute emission potentials
        emission_potential = 0
        for t in range(len(y)):
            emission_potential += np.dot(x[t], feature_weights[:, y[t]])
        
        # Compute transition potentials
        transition_potential = 0
        for t in range(1, len(y)):
            transition_potential += transition_weights[y[t-1], y[t]]
            
        return emission_potential + transition_potential
    
    def forward(self, x):
        """
        Run forward algorithm to compute alpha values
        
        Parameters:
        -----------
        x : numpy.ndarray
            Feature matrix of shape (sequence_length, num_features)
        
        Returns:
        --------
        numpy.ndarray
            Alpha values of shape (sequence_length, num_labels)
        """
        feature_weights = self._get_feature_weights()
        transition_weights = self._get_transition_weights()
        
        sequence_length = len(x)
        # Initialize alpha (forward variables)
        alpha = np.zeros((sequence_length, self.num_labels))
        
        # Base case: t=0
        for label in range(self.num_labels):
            alpha[0, label] = np.dot(x[0], feature_weights[:, label])
        
        # Recursive case
        for t in range(1, sequence_length):
            for current_label in range(self.num_labels):
                # Emission score
                emission_score = np.dot(x[t], feature_weights[:, current_label])
                
                # Transition scores from all previous labels
                max_score = -np.inf
                sum_exp_scores = 0
                
                for prev_label in range(self.num_labels):
                    score = alpha[t-1, prev_label] + transition_weights[prev_label, current_label]
                    max_score = max(max_score, score)
                
                # Log-sum-exp trick for numerical stability
                for prev_label in range(self.num_labels):
                    score = alpha[t-1, prev_label] + transition_weights[prev_label, current_label]
                    sum_exp_scores += np.exp(score - max_score)
                
                alpha[t, current_label] = emission_score + max_score + np.log(sum_exp_scores)
        
        return alpha
    
    def backward(self, x):
        """
        Run backward algorithm to compute beta values
        
        Parameters:
        -----------
        x : numpy.ndarray
            Feature matrix of shape (sequence_length, num_features)
        
        Returns:
        --------
        numpy.ndarray
            Beta values of shape (sequence_length, num_labels)
        """
        feature_weights = self._get_feature_weights()
        transition_weights = self._get_transition_weights()
        
        sequence_length = len(x)
        # Initialize beta (backward variables)
        beta = np.zeros((sequence_length, self.num_labels))
        
        # Base case: t=T-1 (last position)
        for label in range(self.num_labels):
            beta[sequence_length-1, label] = 0  # Log of 1
        
        # Recursive case (going backwards)
        for t in range(sequence_length-2, -1, -1):
            for current_label in range(self.num_labels):
                # Compute scores for all possible next labels
                max_score = -np.inf
                sum_exp_scores = 0
                
                for next_label in range(self.num_labels):
                    emission_score = np.dot(x[t+1], feature_weights[:, next_label])
                    score = beta[t+1, next_label] + transition_weights[current_label, next_label] + emission_score
                    max_score = max(max_score, score)
                
                # Log-sum-exp trick for numerical stability
                for next_label in range(self.num_labels):
                    emission_score = np.dot(x[t+1], feature_weights[:, next_label])
                    score = beta[t+1, next_label] + transition_weights[current_label, next_label] + emission_score
                    sum_exp_scores += np.exp(score - max_score)
                
                beta[t, current_label] = max_score + np.log(sum_exp_scores)
        
        return beta
    
    def compute_marginals(self, x):
        """
        Compute marginal probabilities for a sequence
        
        Parameters:
        -----------
        x : numpy.ndarray
            Feature matrix of shape (sequence_length, num_features)
        
        Returns:
        --------
        tuple
            (node_marginals, edge_marginals)
        """
        feature_weights = self._get_feature_weights()
        transition_weights = self._get_transition_weights()
        
        sequence_length = len(x)
        alpha = self.forward(x)
        beta = self.backward(x)
        
        # Compute log partition function
        log_z = self.compute_partition_function(x)
        
        # Node marginals (probability of label at each position)
        node_marginals = np.zeros((sequence_length, self.num_labels))
        for t in range(sequence_length):
            for label in range(self.num_labels):
                node_marginals[t, label] = np.exp(alpha[t, label] + beta[t, label] - log_z)
        
        # Edge marginals (probability of label transition)
        edge_marginals = np.zeros((sequence_length-1, self.num_labels, self.num_labels))
        for t in range(sequence_length-1):
            for prev_label in range(self.num_labels):
                for current_label in range(self.num_labels):
                    if t == 0:
                        prev_alpha = np.dot(x[0], feature_weights[:, prev_label])
                    else:
                        prev_alpha = alpha[t, prev_label]
                    
                    current_emission = np.dot(x[t+1], feature_weights[:, current_label])
                    current_beta = beta[t+1, current_label]
                    transition = transition_weights[prev_label, current_label]
                    
                    edge_marginals[t, prev_label, current_label] = np.exp(
                        prev_alpha + transition + current_emission + current_beta - log_z
                    )
        
        return node_marginals, edge_marginals
    
    def compute_partition_function(self, x):
        """
        Compute the partition function for a sequence using the forward algorithm.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Feature matrix of shape (sequence_length, num_features)
        
        Returns:
        --------
        float
            Log of the partition function
        """
        alpha = self.forward(x)
        
        # Final partition function is the log-sum of the final alpha values
        max_final = np.max(alpha[-1])
        log_partition = max_final + np.log(np.sum(np.exp(alpha[-1] - max_final)))
        
        return log_partition
    
    def negative_log_likelihood(self, weights, X, Y):
        """
        Compute the negative log likelihood of the data
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Model weights
        X : list of numpy.ndarray
            List of feature matrices
        Y : list of numpy.ndarray
            List of label sequences
            
        Returns:
        --------
        float
            Negative log likelihood
        """
        self.weights = weights
        nll = 0
        
        for x, y in zip(X, Y):
            # Add potential of the true sequence
            sequence_potential = self.compute_potential(x, y)
            
            # Subtract log partition function
            sequence_partition = self.compute_partition_function(x)
            
            # Compute the negative log-likelihood for this sequence
            sequence_nll = -(sequence_potential - sequence_partition)
            nll += sequence_nll
            
        # Add L2 regularization
        reg_strength = 0.1  # Regularization strength
        l2_term = reg_strength * 0.5 * np.sum(self.weights ** 2)
        
        return nll + l2_term
    
    def gradient(self, weights, X, Y):
        """
        Compute the gradient of the negative log likelihood
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Model weights
        X : list of numpy.ndarray
            List of feature matrices
        Y : list of numpy.ndarray
            List of label sequences
            
        Returns:
        --------
        numpy.ndarray
            Gradient of the negative log likelihood
        """
        self.weights = weights
        grad = np.zeros_like(weights)
        feature_weights_shape = (self.num_features, self.num_labels)
        transition_weights_shape = (self.num_labels, self.num_labels)
        
        for x, y in zip(X, Y):
            # Gradient for empirical expectations (negative part)
            # For feature weights
            for t in range(len(y)):
                for f in range(self.num_features):
                    if x[t, f] != 0:  # Skip if feature is zero
                        grad[y[t] * self.num_features + f] -= x[t, f]
            
            # For transition weights
            for t in range(1, len(y)):
                transition_idx = self.num_features * self.num_labels + y[t-1] * self.num_labels + y[t]
                grad[transition_idx] -= 1
            
            # Gradient for model expectations (positive part)
            node_marginals, edge_marginals = self.compute_marginals(x)
            
            # For feature weights
            for t in range(len(x)):
                for label in range(self.num_labels):
                    for f in range(self.num_features):
                        if x[t, f] != 0:  # Skip if feature is zero
                            grad[label * self.num_features + f] += x[t, f] * node_marginals[t, label]
            
            # For transition weights
            for t in range(len(x) - 1):
                for prev_label in range(self.num_labels):
                    for current_label in range(self.num_labels):
                        transition_idx = self.num_features * self.num_labels + prev_label * self.num_labels + current_label
                        grad[transition_idx] += edge_marginals[t, prev_label, current_label]
        
        # Add L2 regularization gradient
        reg_strength = 0.1  # Should match the one in negative_log_likelihood
        grad += reg_strength * weights
        
        return grad
    
    def fit(self, X, Y, max_iter=100):
        """
        Train the CRF model
        
        Parameters:
        -----------
        X : list of numpy.ndarray
            List of feature matrices
        Y : list of numpy.ndarray
            List of label sequences
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        self
        """
        print(f"Training CRF model with {len(X)} sequences...")
        
        # Start with random weights
        initial_weights = np.random.randn(self.num_features * self.num_labels + self.num_labels * self.num_labels)
        
        # Use L-BFGS algorithm for optimization
        result = minimize(
            fun=self.negative_log_likelihood,
            x0=initial_weights,
            args=(X, Y),
            method='L-BFGS-B',
            jac=self.gradient,
            options={'maxiter': max_iter, 'disp': True}
        )
        
        self.weights = result.x
        print(f"Training completed with final loss: {result.fun}")
        return self
    
    def viterbi_decode(self, x):
        """
        Predict the most likely label sequence using the Viterbi algorithm
        
        Parameters:
        -----------
        x : numpy.ndarray
            Feature matrix of shape (sequence_length, num_features)
            
        Returns:
        --------
        numpy.ndarray
            Predicted sequence of labels
        """
        feature_weights = self._get_feature_weights()
        transition_weights = self._get_transition_weights()
        
        sequence_length = len(x)
        
        # Initialize Viterbi variables
        viterbi = np.zeros((sequence_length, self.num_labels))
        backpointers = np.zeros((sequence_length, self.num_labels), dtype=int)
        
        # Base case: t=0
        for label in range(self.num_labels):
            viterbi[0, label] = np.dot(x[0], feature_weights[:, label])
        
        # Recursive case
        for t in range(1, sequence_length):
            for current_label in range(self.num_labels):
                # Emission score
                emission_score = np.dot(x[t], feature_weights[:, current_label])
                
                # Find the best previous label
                max_score = -np.inf
                best_prev_label = -1
                
                for prev_label in range(self.num_labels):
                    score = viterbi[t-1, prev_label] + transition_weights[prev_label, current_label]
                    if score > max_score:
                        max_score = score
                        best_prev_label = prev_label
                
                viterbi[t, current_label] = emission_score + max_score
                backpointers[t, current_label] = best_prev_label
        
        # Backtrack to find the best path
        best_path = np.zeros(sequence_length, dtype=int)
        best_path[-1] = np.argmax(viterbi[-1])
        
        for t in range(sequence_length - 2, -1, -1):
            best_path[t] = backpointers[t+1, best_path[t+1]]
        
        return best_path
    
    def predict(self, X):
        """
        Predict label sequences for all input sequences
        
        Parameters:
        -----------
        X : list of numpy.ndarray
            List of feature matrices
            
        Returns:
        --------
        list of numpy.ndarray
            List of predicted label sequences
        """
        return [self.viterbi_decode(x) for x in X]
    
    def evaluate(self, X_test, Y_test):
        """
        Evaluate the model's performance
        
        Parameters:
        -----------
        X_test : list of numpy.ndarray
            List of test feature matrices
        Y_test : list of numpy.ndarray
            List of test label sequences
            
        Returns:
        --------
        dict
            Dictionary containing precision, recall and F1 score
        """
        Y_pred = self.predict(X_test)
        
        # Flatten the sequences for evaluation
        y_true = np.concatenate(Y_test)
        y_pred = np.concatenate(Y_pred)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

# Create a synthetic dataset for demonstration
def create_synthetic_data():
    """
    Create a synthetic NER dataset
    
    Returns:
    --------
    tuple
        (sentences, pos_tags, tags)
    """
    sentences = [
        "John lives in New York City",
        "Mary works for Google in California",
        "The conference will be held in Paris next month",
        "Amazon announced their new office in Seattle",
        "President Biden visited Berlin last week"
    ]
    
    pos_tags = [
        ["NNP", "VBZ", "IN", "NNP", "NNP", "NNP"],
        ["NNP", "VBZ", "IN", "NNP", "IN", "NNP"],
        ["DT", "NN", "MD", "VB", "VBN", "IN", "NNP", "JJ", "NN"],
        ["NNP", "VBD", "PRP$", "JJ", "NN", "IN", "NNP"],
        ["NNP", "NNP", "VBD", "NNP", "JJ", "NN"]
    ]
    
    tags = [
        ["B-PER", "O", "O", "B-LOC", "I-LOC", "I-LOC"],
        ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"],
        ["O", "O", "O", "O", "O", "O", "B-LOC", "O", "O"],
        ["B-ORG", "O", "O", "O", "O", "O", "B-LOC"],
        ["B-PER", "I-PER", "O", "B-LOC", "O", "O"]
    ]
    
    return sentences, pos_tags, tags

# Process data for CRF
def process_data_for_crf(sentences, pos_tags, tags):
    """
    Convert text data to feature matrices and label sequences
    
    Parameters:
    -----------
    sentences : list
        List of sentence strings
    pos_tags : list
        List of POS tag lists
    tags : list
        List of NER tag lists
        
    Returns:
    --------
    tuple
        (X_features, y_sequences, tag_to_idx, idx_to_tag)
    """
    # Build vocabulary
    word_set = set()
    tag_set = set()
    pos_set = set()
    
    for sentence in sentences:
        for word in sentence.split():
            word_set.add(word.lower())
    
    for tag_list in tags:
        for tag in tag_list:
            tag_set.add(tag)

    for pos_list in pos_tags:
        for pos in pos_list:
            pos_set.add(pos)
    

    word_to_idx = {word: i+1 for i, word in enumerate(word_set)}
    pos_to_idx = {pos: i+1 for i, pos in enumerate(pos_set)}
    tag_to_idx = {tag: i for i, tag in enumerate(tag_set)}
    idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
    
    X_features = []
    y_sequences = []
    print("checkmark 2")

    for i, sentence in enumerate(sentences):
        print(i,sentence)
        words = sentence.split()
        sentence_pos = pos_tags[i]
        sentence_tags = tags[i]
        
        num_features = len(word_to_idx) + len(pos_to_idx) + 4  # Add 4 for binary features
        feature_matrix = np.zeros((len(words), num_features))
        print("checkmark insider",i)

        for j, word in enumerate(words):
            # One-hot word encoding
            if word.lower() in word_to_idx:
                feature_matrix[j, word_to_idx[word.lower()]] = 1.0
            
            # One-hot POS encoding
            if j < len(sentence_pos) and sentence_pos[j] in pos_to_idx:
                feature_matrix[j, len(word_to_idx) + pos_to_idx[sentence_pos[j]]] = 1.0
            
            # Binary features
            feature_idx = len(word_to_idx) + len(pos_to_idx)
            feature_matrix[j, feature_idx] = 1.0 if word[0].isupper() else 0.0  # Capitalization
            feature_matrix[j, feature_idx + 1] = 1.0 if any(c.isdigit() for c in word) else 0.0  # Has digit
            feature_matrix[j, feature_idx + 2] = 1.0 if j == 0 else 0.0  # First word
            feature_matrix[j, feature_idx + 3] = 1.0 if j == len(words) - 1 else 0.0  # Last word
        

        tag_indices = np.array([tag_to_idx[tag] for tag in sentence_tags])
 
        X_features.append(feature_matrix)
        y_sequences.append(tag_indices)
    
    return X_features, y_sequences, tag_to_idx, idx_to_tag

def main():
    print("Loading NER dataset...")

    train_data = pd.read_csv('data/ner_train.csv')
    test_data = pd.read_csv('data/ner_test.csv')

    train_sentences = []
    train_pos_tags = []
    train_tags = []
    
    for _, row in train_data.iterrows():

        train_sentences.append(row['Sentence'])
        
        train_pos_tags.append(eval(row['POS']))
        
        train_tags.append(eval(row['Tag']) )


    test_sentences = []
    test_pos_tags = []
    test_tags = []
    
    for _, row in test_data.iterrows():
        sentences = row['Sentence']
        test_sentences.append(sentences)
        test_pos_tags.append(eval(row['POS']))
        test_tags.append(eval(row['Tag']))
    
    print("Processing data for CRF...")
    X_train, y_train, tag_to_idx, idx_to_tag = process_data_for_crf(train_sentences, train_pos_tags, train_tags)

    print("done with training")
    X_test, y_test, _, _ = process_data_for_crf(test_sentences, test_pos_tags, test_tags)
    
    
    # Get feature and label dimensions
    num_features = X_train[0].shape[1]
    print(num_features)
    num_labels = len(tag_to_idx)
    print(num_labels)
    print(f"Number of features: {num_features}")
    print(f"Number of labels: {num_labels}")
    
    # Initialize and train CRF model
    crf_model = CRF(num_features=num_features, num_labels=num_labels)
    
    # Train the model with fewer iterations for demonstration
    crf_model.fit(X_train, y_train, max_iter=20)
    
    # Evaluate the model
    evaluation = crf_model.evaluate(X_test, y_test)
    print("\nModel Performance:")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F1 Score: {evaluation['f1']:.4f}")
    
    # Make predictions on test set
    y_pred = crf_model.predict(X_test)
    
    # Print some example predictions
    print("\nExample predictions:")
    for i in range(min(3, len(X_test))):
        pred_tags = [idx_to_tag[idx] for idx in y_pred[i]]
        true_tags = [idx_to_tag[idx] for idx in y_test[i]]
        
        # Get the original sentence (approximately)
        sent_idx = i  # This is just an approximation since we've split the data
        if sent_idx < len(sentences):
            print(f"Sentence: {sentences[sent_idx]}")
            print(f"True tags: {true_tags}")
            print(f"Predicted tags: {pred_tags}")
            print()

if __name__ == "__main__":
    main()