from sklearn.naive_bayes import BernoulliNB
import numpy as np
from stack_exchange_tags.iter_message import IterMessage
import tables

class NaiveBayes:
    def __init__(self, n_tags):

        self.n_tags = n_tags
        self.model = [BernoulliNB()] * self.n_tags

    def fit(self, x_train, y_train):

        # Fit the Naive Bayes Method
        self.model = []

        m = IterMessage(self.n_tags, 'calibrated', 300)

        for tag in range(self.n_tags):
            self.model.append(BernoulliNB())
            self.model[tag].fit(x_train, y_train[:, tag])

            # If the index is evenly divisible by 200, print a message
            m.print_message(tag)


    def predict(self, x):

        t_rows = x.shape[0]
        m = IterMessage(self.n_tags, 'calculated', 300)
        y_predict = np.empty((t_rows, self.n_tags))
        y_prob = np.empty((t_rows, self.n_tags))

        for tag in range(self.n_tags):

            # Predict
            y_predict[:, tag] = self.model[tag].predict(x)

            # Probabilities
            prob = self.model[tag].predict_log_proba(x)
            y_prob[:, tag] = np.append(y_prob, prob[:, 1], axis=1)

            # If the index is evenly divisible by 200, print a message
            m.print_message(tag)

        # Make sure that we get at least one tag
        max_prob = np.argmax(y_prob, axis=1)

        for r in range(y_predict.shape[0]):
            y_predict[r, max_prob[r]] = 1

        return y_predict