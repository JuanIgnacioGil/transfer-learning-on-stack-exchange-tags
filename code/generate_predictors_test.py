import json
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from stack_exchange_tags.generate_wordlists import GenerateWordLists as Wl
from stack_exchange_tags.iter_message import IterMessage
import os

# Input data files are available in the "../data/" directory.
physicsTable = pd.read_csv('../data/test.csv', header=0, index_col='id')

# clean data using BeautifulSoup and Regex. First define functions:
clean_titles = []
clean_content = []


print("Cleaning and parsing the training set StackExchange questions...\n")

# Generate dict with everything
data = []

num_records = physicsTable.shape[0]
m = IterMessage(num_records, 'processed', 1000)

for record, nr in zip(physicsTable.index, range(num_records)):

    this_record = {
        'id': str(record),
        'topic': 'physics',
        'title': Wl.noun_list(physicsTable['title'][record]),
        'content': Wl.noun_list(physicsTable['content'][record]),
        'tags': []
    }

    data.append(this_record)

    # If the index is evenly divisible by 1000, print a message
    m.print_message(nr)

# Convert to json

# Remove the output file if there is an old one
json_file = '../data/test.json'

try:
    os.remove(json_file)
except OSError:
    pass

with open(json_file, 'w') as outfile:
    json.dump(data, outfile, sort_keys=True, indent=4)

