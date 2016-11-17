import json
import time
import deepdish as dd
import scipy.sparse as sp
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from generate_wordlists import *

hfile = '../data/test.h5'

# Input data files are available in the "../data/" directory.
physicsTable = pd.read_csv('../data/test.csv', header=0, index_col='id')

# clean data using BeautifulSoup and Regex. First define functions:
clean_titles = []
clean_content = []


print("Cleaning and parsing the training set StackExchange questions...\n")

# Generate dict with everything
data = []

num_records = physicsTable.shape[0]
for record in physicsTable.index:

    this_record = {
        'id': str(record),
        'topic': 'physics',
        'title': titles_to_wordlist(physicsTable['title'][record], remove_stopwords=True),
        'content':  content_to_wordlist(physicsTable['content'][record], remove_stopwords=True),
        'tags': []
    }

    data.append(this_record)

    # If the index is evenly divisible by 1000, print a message
    if (record + 1) % 1000 == 0:
        print("Question %d of %d\n" % (record + 1, num_records))

# Convert to json
with open('../data/test.json', 'w') as outfile:
    json.dump(data, outfile, sort_keys=True, indent=4)

# Read the data
l_data = len(data)


# List of all possible words in titles
all_title_list = []

for x in data:
    all_title_list += x['title']


all_title = set(all_title_list)
l_title = len(all_title)

# List of all possible words in content
all_content_list = []

for x in data:
    all_content_list += x['content']


all_content = set(all_content_list)
l_content = len(all_content)


# For each row, generate a vector of binary outputs (one for each tag)
# and of binary predictors (one for word in title, one for word in content)
t0 = time.time()
predictors = sp.lil_matrix((l_data, l_title+l_content))


for x, nd in zip(data, range(l_data)):
# for x, nd in zip(data[:100], range(100)):

    for w, n in zip(all_title, range(l_title)):
        if w in x['title']:
            predictors[nd, n] = 1

    for w, n in zip(all_content, range(l_content)):
        if w in x['content']:
            predictors[nd, l_title + n] = 1

    # If the index is evenly divisible by 500, print a message
    if (nd + 1) % 500 == 0:
        p = int((100 * (nd + 1) / l_data))
        elapsed = time.time() - t0
        remaining = int(elapsed * (l_data - nd - 1) / (60 * (nd + 1)))
        print('{}% calculated. {} minutes remaining'.format(p, remaining))


dd.io.save(hfile, {
    'predictors': predictors.tocsr(),
    'title': list(all_title),
    'content': list(all_content)
    })
