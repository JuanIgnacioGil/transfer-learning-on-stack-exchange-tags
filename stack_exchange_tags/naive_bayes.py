import time
from sklearn.naive_bayes import BernoulliNB
import numpy as np


class NaiveBayes:
    def __init__(self, n_tags):

        self.n_tags = n_tags
        self.model = [BernoulliNB()] * self.n_tags

    def fit(self, x_train, y_train):

        # Fit the Naive Bayes Method
        self.model = []
        t0 = time.time()

        for tag in range(self.n_tags):
            self.model.append(BernoulliNB())
            self.model[tag].fit(x_train, y_train[:, tag])

            # If the index is evenly divisible by 200, print a message
            if (tag + 1) % 200 == 0:
                p = int((100 * (tag + 1) / self.n_tags))
                elapsed = time.time() - t0
                remaining = int(elapsed * (self.n_tags - tag - 1) / (60 * (tag + 1)))
                print('{}% calibrated. {} minutes remaining'.format(p, remaining))


    def predict(self, x):

        t_rows = x.shape[0]
        t0 = time.time()
        y_predict = np.zeros((t_rows, self.n_tags))
        y_prob = np.zeros((t_rows, self.n_tags))

        for tag in range(self.n_tags):

            # Predict
            y_predict[:, tag] = self.model[tag].predict(x)

            # Probabilities
            prob = self.model[tag].predict_log_proba(x)
            y_prob[:, tag] = prob[:, 1]

            # If the index is evenly divisible by 200, print a message
            if (tag + 1) % 200 == 0:
                p = int((100 * (tag + 1) / self.n_tags))
                elapsed = time.time() - t0
                remaining = int(elapsed * (self.n_tags - tag - 1) / (60 * (tag + 1)))
                print('{}% calculated. {} minutes remaining'.format(p, remaining))

        # Make sure that we get at least one tag
        max_prob = np.argmax(y_prob, axis=1)

        for r in range(y_predict.shape[0]):
            y_predict[r, max_prob[r]] = 1

        return y_predict