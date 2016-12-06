# Python 3
# On first several passes, just try the simplest things and we'll add complexity later

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from stack_exchange_tags.generate_wordlists import GenerateWordLists as Wl
from stack_exchange_tags.iter_message import IterMessage
import os


# Input data files are available in the "../input/" directory.
biologyTable = pd.read_csv('../data/biology.csv', header=0)
cookingTable = pd.read_csv('../data/cooking.csv', header=0)
cryptoTable = pd.read_csv('../data/crypto.csv', header=0)
diyTable = pd.read_csv('../data/diy.csv', header=0)
roboticsTable = pd.read_csv('../data/robotics.csv', header=0)
travelTable = pd.read_csv('../data/travel.csv', header=0)
physicsTable = pd.read_csv('../data/test.csv', header=0)

json_file = '../data/data.json'

# Remove the output file if there is an old one
try:
    os.remove(json_file)
except OSError:
    pass


# clean data using BeautifulSoup and Regex. First define functions:

clean_titles = []
clean_content = []
clean_tags = []

# form one large data frame with all the training set info to make text cleaning easier
learningTables = [biologyTable, cookingTable, cryptoTable, diyTable, roboticsTable, travelTable]

print("Cleaning and parsing the training set StackExchange questions...\n")

# Generate dict with everything
data = []
topics = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']

for table, top in zip(learningTables, topics):
    num_records = table.shape[0]
    m = IterMessage(num_records, 'processed', 1000)

    for record, nr in zip(table.index, range(num_records)):

        this_record = {
            'topic': top,
            'title': Wl.noun_list(table["title"][record]),
            'content':  Wl.noun_list(table["content"][record]),
            'tags': Wl.tags_to_wordlist(table["tags"][record])
        }

        data.append(this_record)

        # If the index is evenly divisible by 1000, print a message
        m.print_message(nr)


# Convert to json
with open(json_file, 'w') as outfile:
    json.dump(data, outfile,sort_keys=True, indent=4)


