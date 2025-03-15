from sklearn.metrics import precision_recall_fscore_support

def evaluate(model, X_test, Y_test):
    y_pred = [model.predict(X) for X in X_test]
    y_true = [y for Y in Y_test for y in Y]  # Flatten labels
    y_pred_flat = [y for Y in y_pred for y in Y]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_flat, average="weighted")
    return {"precision": precision, "recall": recall, "f1-score": f1}

# Example evaluation
metrics = evaluate(crf, X_train, Y_train)
print(metrics)
