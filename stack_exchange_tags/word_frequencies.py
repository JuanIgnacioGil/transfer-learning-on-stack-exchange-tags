import json
import deepdish as dd
from stack_exchange_tags.iter_message import IterMessage
import os
import nltk

class WordFrequencies:

    def __init__(self, **kwargs):
        self.hfile = kwargs.get('hfile')
        self.json_data = kwargs.get('json_data')

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
        l_tags = len(all_tags)

        # List of all possible words in titles
        all_title_list = []

        for x in data:
            all_title_list += x['title']


        all_title = nltk.FreqDist(all_title_list)
        l_title = len(all_title)

        # List of all possible words in content
        all_content_list = []

        for x in data:
            all_content_list += x['content']


        all_content = nltk.FreqDist(all_content_list)
        l_content = len(all_content)


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

    def higher_frequency_as_tag():
