## Problem :

Derive the mathematical formulation of a linear-chain CRF. 
Implement the CRF model using Python (e.g., NumPy and SciPy). 
Train the model on a provided dataset. 
Evaluate the model’s performance using metrics such as precision, recall, and F1-score.


## Flow :

[Features(POS Tags, words etc) extractions] -> initalize weights -> use SGD/BFSD to adjust weights -> Infer using DP(Viterbi) -> Evaluate Results





## Report:

    1. Introduction 
    2.1 CRF Explanation 
    2.2 CRF Derivation 
    3. Feature Extraction Techniques used
    4.1 Gradient Descent and epochs , results at each epoch, algorithms used to do gradient descent
    4.2 Store weight ? .pkl /wandb
    5.1 Inference and evaluation metrics in a tabular format
    5.2 evaluation metrics used ( Precision, Recall , F1).
    6. Future work 
    7. Citations and Acknowledgement 


## Resources : 

https://www.youtube.com/watch?v=8AnHsHhiJ4U
https://www.youtube.com/watch?v=7CRyqwCZFY0
https://web.stanford.edu/~jurafsky/asru09.pdf
https://web.stanford.edu/~jurafsky/slp3/17.pdf
https://medium.com/data-science-in-your-pocket/named-entity-recognition-ner-using-conditional-random-fields-in-nlp-3660df22e95c# crf_nlp



Data exploration:

dataset size


Number of sentences      = 35177
Number of unique words:  = 30172
Number of unique tags :  = 9
unique tags           :  = ['gpe', 'tim', 'art', 'nat', 'geo', 'per', 'eve', 'org', 'O']

[show a plot here]

![data.head()](image.png)

show the frequency of each tag :

{'O': 889973,
 'geo': 44934,
 'gpe': 16621,
 'per': 34393,
 'org': 36721,
 'tim': 26491,
 'art': 714,
 'nat': 302,
 'eve': 645}



Feature Engineering :

Create features using words :

Feature : { 
    "word": "Apple",
    "pos": "NNP",
    "is_first": True,
    "is_last": False,
    "prev_word": None,
    "next_word": "is",
    "prev_pos": None,
    "next_pos": "VBZ",
    "is_title": True,
}