import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from nb import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X, y = datasets.make_classification(n_samples=20, n_features=5, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =123)


print(X_train)


nb = NaiveBayes()

nb.fit(X_train, y_train)

prediction = nb.predict(X_test)
print(prediction)

print("Naive bayes classification accuracy", accuracy(y_test, prediction))


