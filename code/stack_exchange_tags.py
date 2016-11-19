from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import deepdish as dd
import time
import numpy as np
import pandas as pd
import os
import csv


class StackExchangeTags:

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(self):
        pass

    @staticmethod
    def validation_sets(train_file):

        # Read data
        x = dd.io.load(train_file, '/predictors')
        y = dd.io.load(train_file, '/outputs').toarray()

        # Generate train and validation
        x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=0)

        return x_train, x_validation, y_train, y_validation

    @staticmethod
    def tests_sets(train_file, test_file):

        # Read data
        x_train = dd.io.load(train_file, '/predictors')
        y_train = dd.io.load(train_file, '/outputs').toarray()
        x_test = dd.io.load(test_file, '/predictors').toarray()

        return x_train, y_train, x_test

    @staticmethod
    def naive_bayes(x_train, y_train, x_test):

        t_rows = x_test.shape[0]
        n_tags = y_train.shape[1]

        # Fit the Naive Bayes Method
        clf = BernoulliNB()
        t0 = time.time()
        y_predict = np.zeros((t_rows, n_tags))
        y_prob = np.zeros((t_rows, n_tags))

        for tag in range(n_tags):
            clf.fit(x_train, y_train[:, tag])

            # Predict
            ytp = clf.predict(x_test)
            ytp[np.isnan(ytp)] = 0
            y_predict[:, tag] = ytp

            # Probabilities
            prob = clf.predict_log_proba(x_test)
            y_prob[:, tag] = prob[:, 1]

            # If the index is evenly divisible by 200, print a message
            if (tag + 1) % 200 == 0:
                p = int((100 * (tag + 1) / n_tags))
                elapsed = time.time() - t0
                remaining = int(elapsed * (n_tags - tag - 1) / (60 * (tag + 1)))
                print('{}% calculated. {} minutes remaining'.format(p, remaining))

        # Make sure that we get at least one tag
        max_prob = np.argmax(y_prob, axis=1)

        for r in range(y_predict.shape[0]):
            y_predict[r, max_prob[r]] = 1

        return y_predict

    @staticmethod
    def validate(y_predict, y_validation, train_file, output_file):

        tags = dd.io.load(train_file, '/tags')

        # Evaluate
        true_positives = 0
        predicted_positives = 0
        actual_positives = 0

        for r in range(y_validation.shape[0]):
            yvr = y_validation[r, :]
            ypr = y_predict[r, :]
            true_positives += sum([a * b for a, b in zip(ypr, yvr)])
            predicted_positives += sum(ypr)
            actual_positives += sum(yvr)

        precision = true_positives / predicted_positives
        recall = true_positives / actual_positives
        f1 = 2 * precision * recall / (precision + recall)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('F1: {}'.format(f1))

        # Generate list of tags for comparing

        # Remove the output file if there is an old one
        try:
            os.remove(output_file)
        except OSError:
            pass

        with open(output_file, 'a') as out:

            for r in range(y_validation.shape[0]):
                predicted_tags = [t for (t, x) in zip(tags, y_predict[r, :]) if int(round(x)) is 1]
                actual_tags = [t for (t, x) in zip(tags, y_validation[r, :]) if int(round(x)) is 1]

                out.write('{} -> {}\n'.format(actual_tags, predicted_tags))

        return f1, precision, recall

    @staticmethod
    def generate_submission(y_predict, train_file, test_csv_file, submission):

        tags = dd.io.load(train_file, '/tags')

        # Load ids from csv file
        # TODO: Store id in h5 file, so that we don't need the csv
        # Read the csv file
        physicsTable = pd.read_csv(test_csv_file, header=0, index_col='id')
        id = [record['id'] for record in physicsTable]

        # Generate list of tags for comparing

        # Remove the output file if there is an old one
        try:
            os.remove(submission)
        except OSError:
            pass

        with open(submission, 'a') as s:

            writer = csv.writer(s, delimiter=',', lineterminator='\r\n', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['id', 'tags'])

            for r in range(y_predict.shape[0]):
                predicted_tags = [t for (t, x) in zip(tags, y_predict[r, :]) if int(round(x)) is 1]
                row_id = id[r]

                writer.writerow([row_id, predicted_tags])

            s.close()
