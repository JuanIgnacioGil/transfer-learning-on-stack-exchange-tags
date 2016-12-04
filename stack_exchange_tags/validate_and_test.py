from sklearn.model_selection import train_test_split
import deepdish as dd
import pandas as pd
import os
import csv
from stack_exchange_tags.naive_bayes import NaiveBayes
from stack_exchange_tags.iter_message import IterMessage
from stack_exchange_tags.generate_wordlists import GenerateWordLists as Wl
import json
import scipy.sparse as sp
import numpy as np


class StackExchangeTags:

    def __init__(self, **kwargs):

        self.train_file = kwargs.get('train_file', '')
        self.validation_file = kwargs.get('validation_file', '')
        self.test_file = kwargs.get('test_file', '')
        self.test_json_file = kwargs.get('test_json_file', '')
        self.test_csv_file = kwargs.get('test_csv_file', '')
        self.submission = kwargs.get('submission', '')
        self.n = kwargs.get('n', 2)

    def validation_sets(self, **kwargs):

        train_file = kwargs.get('train_file', self.train_file)

        # Read data
        x = dd.io.load(train_file, '/predictors')
        y = np.array(dd.io.load(train_file, '/outputs'))

        # Generate train and validation
        x_train, x_validation, y_train, y_validation = train_test_split(x, y[3:], test_size=0.25, random_state=0)

        return x_train, x_validation, y_train, y_validation

    def tests_sets(self, **kwargs):

        train_file = kwargs.get('train_file', self.train_file)
        test_file = kwargs.get('test_file', self.test_file)

        # Read data
        x_train = dd.io.load(train_file, '/predictors')
        y_train = dd.io.load(train_file, '/outputs').toarray()
        x_test = dd.io.load(test_file, '/predictors').toarray()

        return x_train, y_train, x_test

    def validate(self, **kwargs):

        train_file = kwargs.get('train_file', self.train_file)
        validation_file = kwargs.get('validation_file', self.validation_file)
        n = kwargs.get('n', self.n)

        tags = dd.io.load(train_file, '/tags')

        # Generate validation sets
        x_train, x_validation, y_train, y_validation = self.validation_sets(train_file=train_file)

        # Fit model
        n_tags = len(y_validation)
        model = NaiveBayes(n_tags)
        model.fit(x_train, y_train)

        # Evaluate
        y_predict = model.predict(x_validation, 1)
        lyv = y_validation.shape[0]
        predicted_positives = len(y_predict)
        actual_positives = lyv + 1
        true_positives = 0

        for r in range(lyv):
            yvr = y_validation[r]
            ypr = y_predict[r]
            true_positives += (yvr == ypr)
            predicted_positives += ypr
            actual_positives += yvr

        precision = true_positives / predicted_positives
        recall = true_positives / actual_positives
        f1 = 2 * precision * recall / (precision + recall)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('F1: {}'.format(f1))

        # Generate list of tags for comparing

        # Remove the output file if there is an old one
        try:
            os.remove(validation_file)
        except OSError:
            pass

        m = IterMessage(lyv, 'tags generated', 300)

        with open(validation_file, 'a') as out:

            for r in range(lyv):
                predicted_tags = tags[y_predict[r]]
                actual_tags = tags[y_validation[r]]

                out.write('{} -> {}\n'.format(actual_tags, predicted_tags))
                m.print_message(r)

        return f1, precision, recall

    def generate_submission(self, **kwargs):

        train_file = kwargs.get('train_file', self.train_file)
        test_file = kwargs.get('test_file', self.test_file)
        test_csv_file = kwargs.get('test_csv_file', self.test_csv_file)
        submission = kwargs.get('submission', self.submission)
        n = kwargs.get('n', self.n)

        tags = dd.io.load(train_file, '/tags')

        # Load ids from csv file
        # TODO: Store id in h5 file, so that we don't need the csv
        # Read the csv file
        physics_table = pd.read_csv(test_csv_file, header=0, index_col='id')
        se_id = physics_table.index

        # Generate train and test sets
        x_train, y_train, x_test = self.tests_sets(train_file=train_file, test_file=test_file)

        # Fit model
        n_tags = y_train.shape[1]
        model = NaiveBayes(n_tags)
        model.fit(x_train, y_train)

        # Remove the output file if there is an old one
        try:
            os.remove(submission)
        except OSError:
            pass

        # Predict
        n_rows = x_test.shape[0]
        y_predict = model.predict(x_test, n)

        with open(submission, 'a') as s:

            m = IterMessage(n_rows, 'tags generated', 1000)
            writer = csv.writer(s, delimiter=',', lineterminator='\r\n', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['id', 'tags'])

            for r in range(n_rows):

                predicted_tags = tags[y_predict[r, :]]
                row_id = str(se_id[r])
                writer.writerow([row_id, predicted_tags])
                m.print_message(r)

            s.close()

    def generate_predictors_test(self, **kwargs):

        hfile = kwargs.get('test_file', self.test_file)
        test_csv_file = kwargs.get('test_csv_file', self.test_csv_file)
        test_json_file = kwargs.get('test_json_file', self.test_json_file)
        train_file = kwargs.get('train_file', self.train_file)

        # Input data files are available in the "../data/" directory.
        physics_table = pd.read_csv(test_csv_file, header=0, index_col='id')

        print("Cleaning and parsing the training set StackExchange questions...\n")

        # Generate dict with everything
        data = []

        num_records = physics_table.shape[0]
        m = IterMessage(num_records, 'processed to json', 1000)

        for record, nr in zip(physics_table.index, range(num_records)):

            this_record = {
                'id': str(record),
                'topic': 'physics',
                'title': Wl.titles_to_wordlist(physics_table['title'][record], remove_stopwords=True),
                'content': Wl.content_to_wordlist(physics_table['content'][record], remove_stopwords=True),
                'tags': []
            }

            data.append(this_record)

            # If the index is evenly divisible by 1000, print a message
            m.print_message(record)

        # Convert to json
        with open(test_json_file, 'w') as outfile:
            json.dump(data, outfile, sort_keys=True, indent=4)

        # Read the data
        l_data = len(data)

        # List of all possible words in titles and content
        all_title = dd.io.load(train_file, '/title')
        all_content = dd.io.load(train_file, '/content')

        l_title = len(all_title)
        l_content = len(all_content)

        # For each row, generate a vector of binary outputs (one for each tag)
        # and of binary predictors (one for word in title, one for word in content)
        predictors = sp.lil_matrix((l_data, l_title + l_content))
        m = IterMessage(l_data, 'predictor rows created', 500)

        for x, nd in zip(data, range(l_data)):
            # for x, nd in zip(data[:100], range(100)):

            for w, n in zip(all_title, range(l_title)):
                if w in x['title']:
                    predictors[nd, n] = 1

            for w, n in zip(all_content, range(l_content)):
                if w in x['content']:
                    predictors[nd, l_title + n] = 1

            # If the index is evenly divisible by 500, print a message
            m.print_message(nd)

        dd.io.save(hfile, {
            'predictors': predictors.tocsr(),

        })


