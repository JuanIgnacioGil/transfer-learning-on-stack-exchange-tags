from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import deepdish as dd
import time
import numpy as np

# Read data
X = dd.io.load('../data/data.h5', '/predictors')
Y = dd.io.load('../data/data.h5', '/outputs')
tags = dd.io.load('../data/data.h5', '/tags')

n_tags = Y.shape[1]

# Generate train and validation
X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.25, random_state=0)


# Fit the Naive Bayes Method
clf = BernoulliNB()
t0 = time.time()
y_predict = np.zeros(y_validation.shape)

for tag in range(n_tags):
    clf.fit(X_train, y_train[:, tag].toarray().ravel())

    # Predict
    ytp = clf.predict(X_validation)
    ytp[np.isnan(ytp)] = 0
    y_predict[:, tag] = ytp

    # If the index is evenly divisible by 200, print a message
    if (tag + 1) % 200 == 0:
        p = int((100 * (tag + 1) / n_tags))
        elapsed = time.time() - t0
        remaining = int(elapsed * (n_tags - tag - 1) / (60 * (tag + 1)))
        print('{}% calculated. {} minutes remaining'.format(p, remaining))

#Generate binary predictions
y_predict_binary = y_predict.copy()
y_predict_binary[y_predict_binary<=0.5]=0
y_predict_binary[y_predict_binary>0.5]=1


# Evaluate
true_positives = 0
predicted_positives = 0
actual_positives = 0


for r in range(y_validation.shape[0]):
    yvr = y_validation[r, :].toarray().ravel()
    true_positives += sum([a * b for a, b in zip(y_predict[r, :], yvr)])
    predicted_positives += sum(y_predict[r, :])
    actual_positives += sum(yvr)

precision = true_positives / predicted_positives
recall = true_positives / actual_positives
F1 = 2 * precision * recall / (precision + recall)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('F1: {}'.format(F1))


# Generate list of tags for comparation
output_file = '../data/naive_bayes_validation_output.txt'

with open(output_file, 'a') as out:

    for r in range(y_validation.shape[0]):
        predicted_tags = [t for (t, x) in zip(tags, y_predict[r, :]) if x is not 0]
        actual_tags = [t for (t, x) in zip(tags, y_validation[r, :].toarray()) if x is not 0]

        out.write('{} -> {}\n'.format(actual_tags, predicted_tags))
