import pandas as pd
import nltk
import os
from stack_exchange_tags.iter_message import IterMessage

class POSAnalysis:

    def __init__(self, hdf5_file, data_folder):

        self.hdf5_file = hdf5_file
        self.data_folder = data_folder


    def generate_hdf5_file(self, topics, key):

        hdf = pd.HDFStore(self.hdf5_file)

        try:
            print(hdf[key])
        except:
            data = pd.DataFrame(columns={'id', 'topic', 'word', 'section', 'POS', 'frequency'})
            hdf.put(key, data, format='table', data_columns=True)



        for topic in ['biology', 'cooking', 'crypto', 'robotics']:

            print(topic)
            csv_file = os.path.join(self.data_folder,'{}.csv'.format(topic))

            # Read csv file
            table = pd.read_csv(csv_file, header=0)
            n_records = len(table.index)
            data = []
            m = IterMessage(n_records, 'records processed', 500)

            #Empty dataframe for the records
            data = pd.DataFrame(columns = {'id', 'topic', 'word', 'section', 'POS', 'frequency'})

            # Generate POS and frequencies
            for record, nr in zip(table.index, range(n_records)):

                id = table['id'][record]

                # title
                title = table['title'][record]

                data = data.append(self.pos_and_frequencies(table['title'][record], id, 'title', topic))
                data = data.append(self.pos_and_frequencies(table['content'][record], id, 'content', topic))
                data = data.append(self.pos_and_frequencies(table['tags'][record], id, 'tags', topic))

                # If the index is evenly divisible by 500, print a message
                m.print_message(nr)

            # Save data
            # Save the results to the file
            hdf.append(key, data, format='table', data_columns=True)

    @classmethod
    def pos_and_frequencies(cls, text, id, section, topic):
        words = nltk.word_tokenize(text.lower())
        pos_text = nltk.pos_tag(words, tagset='universal')
        freq = nltk.ConditionalFreqDist(pos_text)

        all_words = freq.conditions()
        output = []

        for w in all_words:
            pos = freq[w].most_common()[0][0]
            frequency = freq[w].most_common()[0][1]
            output.append([id, topic, w, section, pos, frequency])

        return pd.DataFrame(output, columns = {'id', 'topic', 'word', 'section', 'POS', 'frequency'})







