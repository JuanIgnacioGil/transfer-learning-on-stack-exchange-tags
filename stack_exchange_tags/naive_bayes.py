from sklearn.naive_bayes import BernoulliNB


class NaiveBayes:
    def __init__(self, n_tags):

        self.n_tags = n_tags
        self.model = BernoulliNB()

    def fit(self, x_train, y_train):

        # Fit the Naive Bayes Method
        self.model.fit(x_train, y_train)

    def predict(self, x, n):

        # Probabilities
        y_prob = self.model.predict_log_proba(x).tolist()

        # Look for the n bigger
        return y_prob.argsort(axis=1)[:, -n:]
