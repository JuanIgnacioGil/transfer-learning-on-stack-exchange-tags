# Python 3
# On first several passes, just try the simplest things and we'll add complexity later

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from generate_wordlists import *

# Input data files are available in the "../input/" directory.
biologyTable = pd.read_csv('../data/biology.csv', header=0)
cookingTable = pd.read_csv('../data/cooking.csv', header=0)
cryptoTable = pd.read_csv('../data/crypto.csv', header=0)
diyTable = pd.read_csv('../data/diy.csv', header=0)
roboticsTable = pd.read_csv('../data/robotics.csv', header=0)
travelTable = pd.read_csv('../data/travel.csv', header=0)
physicsTable = pd.read_csv('../data/test.csv', header=0)


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
    for record in table.index:

        this_record = {
            'topic': top,
            'title': titles_to_wordlist(table["title"][record], remove_stopwords=True),
            'content':  content_to_wordlist(table["content"][record], remove_stopwords=True),
            'tags': tags_to_wordlist(table["tags"][record])
        }

        data.append(this_record)

        # If the index is evenly divisible by 1000, print a message
        if (record + 1) % 1000 == 0:
            print("Question %d of %d\n" % (record + 1, num_records))

# Convert to json
with open('../data/data.json', 'w') as outfile:
    json.dump(data, outfile,sort_keys=True, indent=4)
