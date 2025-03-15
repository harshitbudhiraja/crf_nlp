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
