import pandas as pd
import nltk
from stack_exchange_tags.iter_message import IterMessage

class POSAnalysis:

    def __init__(self, hdf5_file):

        self.hdf5_file = hdf5_file


    def generate_hdf5_file(self, topic, csv_file):

        # Read csv file
        table = pd.read_csv(csv_file, header=0)
        n_records = len(table.index)
        data = []
        m = IterMessage(n_records, 'records processed', 500)

        # Generate POS and frequencies
        for record, nr in zip(table.index, range(n_records)):

            this_record = dict()

            this_record['title'] = self.pos_and_frequencies(table['title'][record])
            this_record['content'] = self.pos_and_frequencies(table['content'][record])
            this_record['tags'] = self.pos_and_frequencies(table['tags'][record])

            data.append(this_record)

            # If the index is evenly divisible by 500, print a message
            m.print_message(nr)

        # Save data
        # Save the results to the file
        store = pd.HDFStore(self.hdf5_file)
        store[topic] = data
        store.close()

    @classmethod
    def pos_and_frequencies(cls, text):
        words = nltk.word_tokenize(text.lower())
        pos_text = nltk.pos_tag(words, tagset='universal')
        freq = nltk.ConditionalFreqDist(pos_text)
        return freq







