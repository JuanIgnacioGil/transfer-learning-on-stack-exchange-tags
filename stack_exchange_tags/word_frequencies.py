import json
import deepdish as dd
from stack_exchange_tags.iter_message import IterMessage
import os
import nltk
import csv
import pandas as pd


class WordFrequencies:

    def __init__(self, **kwargs):
        self.hfile = kwargs.get('hfile')
        self.json_data = kwargs.get('json_data')
        self.submission = kwargs.get('submission')
        self.validation_file = kwargs.get('validation_file')
        self.test_hfile = kwargs.get('test_hfile')
        self.test_json = kwargs.get('test_json')
        self.test_csv_file = kwargs.get('test_csv_file')

    def generate_word_frequencies(self, **kwargs):

        hfile = kwargs.get('hfile', self.hfile)
        json_data = kwargs.get('json_data', self.json_data)

        # Remove the output file if there is an old one
        try:
            os.remove(hfile)
        except OSError:
            pass

        # Read the json file
        with open(json_data):
            data = json.load(json_data)
            json_data.close()

        l_data = len(data)

        # Generate a list of all possible tags
        all_tags_list = []

        for x in data:
            all_tags_list += x['tags']

        all_tags = nltk.FreqDist(all_tags_list)

        # List of all possible words in titles
        all_title_list = []

        for x in data:
            all_title_list += x['title']

        all_title = nltk.FreqDist(all_title_list)

        # List of all possible words in content
        all_content_list = []

        for x in data:
            all_content_list += x['content']

        all_content = nltk.FreqDist(all_content_list)

        # For each tag, generate ann output and a vector of binary predictors
        # (one for word in title, one for word in content)
        m = IterMessage(l_data, 'processed', 500)
        frequencies = []

        for x, nd in zip(data, range(l_data)):

            this_frequencies = dict()
            this_frequencies['title'] = nltk.FreqDist(x['title'])
            this_frequencies['content'] = nltk.FreqDist(x['content'])
            this_frequencies['all'] = nltk.FreqDist(x['title'] + x['content'])
            this_frequencies['tags'] = x['tags']

            frequencies.append(this_frequencies)

            # If the index is evenly divisible by 500, print a message
            m.print_message(nd)

        # Save the results to the file
        dd.io.save(hfile, {
            'frequencies': frequencies,
            'tags': all_tags,
            'title': all_title,
            'content': all_content
            })

    def generate_word_test(self, **kwargs):

        test_hfile = kwargs.get('test_hfile', self.test_hfile)
        test_json = kwargs.get('test_json', self.test_json)

        # Remove the output file if there is an old one
        try:
            os.remove(test_hfile)
        except OSError:
            pass

        # Read the json file
        with open(test_json) as t:
            data = json.load(t)
            t.close()

        l_data = len(data)

        # List of all possible words in titles
        all_title_list = []

        for x in data:
            all_title_list += x['title']

        all_title = nltk.FreqDist(all_title_list)

        # List of all possible words in content
        all_content_list = []

        for x in data:
            all_content_list += x['content']

        all_content = nltk.FreqDist(all_content_list)

        # For each tag, generate ann output and a vector of binary predictors
        # (one for word in title, one for word in content)
        m = IterMessage(l_data, 'processed', 500)
        frequencies = []

        for x, nd in zip(data, range(l_data)):

            this_frequencies = dict()
            this_frequencies['title'] = nltk.FreqDist(x['title'])
            this_frequencies['content'] = nltk.FreqDist(x['content'])
            this_frequencies['all'] = nltk.FreqDist(x['title'] + x['content'])

            frequencies.append(this_frequencies)

            # If the index is evenly divisible by 500, print a message
            m.print_message(nd)

        # Save the results to the file
        dd.io.save(test_hfile, {
            'frequencies': frequencies,
            'title': all_title,
            'content': all_content
            })

    def validate(self, **kwargs):

        train_file = kwargs.get('hfile', self.hfile)
        validation_file = kwargs.get('validation_file', self.validation_file)

        # Read data
        frequencies = dd.io.load(train_file, '/frequencies')

        # Remove the output file if there is an old one
        try:
            os.remove(validation_file)
        except OSError:
            pass

        # Validate

        predicted_positives = 0
        actual_positives = 0
        true_positives = 0

        lf = len(frequencies)
        m = IterMessage(lf, 'tags generated', 1000)

        with open(validation_file, 'a') as vf:

            for x, r in zip(frequencies, range(lf)):
                actual_tags = x['tags']
                words = x['all'].most_common(2)[:2]

                predicted_tags = [k[0] for k in words]

                predicted_positives += 2
                actual_positives += len(actual_tags)

                vf.write('{} -> {}\n'.format(actual_tags, predicted_tags))
                m.print_message(r)

                for t in actual_tags:
                    if t in predicted_tags:
                        true_positives += 1

        precision = true_positives / predicted_positives
        recall = true_positives / actual_positives
        f1 = 2 * precision * recall / (precision + recall)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('F1: {}'.format(f1))

        return f1, precision, recall

    def generate_submission(self, **kwargs):

        train_file = kwargs.get('test_hfile', self.test_hfile)
        submission = kwargs.get('submission', self.submission)
        test_csv_file = kwargs.get('test_csv_file', self.test_csv_file)

        # Read data
        frequencies = dd.io.load(train_file, '/frequencies')

        # Read the csv file
        physics_table = pd.read_csv(test_csv_file, header=0, index_col='id')
        se_id = physics_table.index

        # Remove the output file if there is an old one
        try:
            os.remove(submission)
        except OSError:
            pass

        # Generate submission

        lf = len(frequencies)
        m = IterMessage(lf, 'tags generated', 1000)

        with open(submission, 'a') as s:

            writer = csv.writer(s, delimiter=',', lineterminator='\r\n', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['id', 'tags'])

            for x, r in zip(frequencies, range(lf)):

                words = x['all'].most_common(2)[:2]
                predicted_tags = [k[0] for k in words]

                row_id = str(se_id[r])
                writer.writerow([row_id, ' '.join(predicted_tags)])
                m.print_message(r)


