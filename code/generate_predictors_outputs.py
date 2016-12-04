import json
import deepdish as dd
import scipy.sparse as sp
from stack_exchange_tags.iter_message import IterMessage
from scipy.sparse import vstack
import os

hfile = '../data/data.h5'

# Remove the output file if there is an old one
try:
    os.remove(hfile)
except OSError:
    pass

# Read the json file
with open('../data/data.json') as json_data:
    data = json.load(json_data)
    json_data.close()

l_data = len(data)

# Generate a list of all possible tags
all_tags_list = []

for x in data:
    all_tags_list += x['tags']


all_tags = list(set(all_tags_list))
l_tags = len(all_tags)

# List of all possible words in titles
all_title_list = []

for x in data:
    all_title_list += x['title']


all_title = list(set(all_title_list))
l_title = len(all_title)

# List of all possible words in content
all_content_list = []

for x in data:
    all_content_list += x['content']


all_content = list(set(all_content_list))
l_content = len(all_content)


# For each tag, generate ann output and a vector of binary predictors
# (one for word in title, one for word in content)
m = IterMessage(l_data, 'processed', 500)
outputs = []

for x, nd in zip(data, range(l_data)):

    predictors_t = sp.lil_matrix((1, l_title + l_content))

    for w in x['title']:
        index = all_title.index(w)
        predictors_t[0, index] = 1

    for w in x['content']:
        index = all_content.index(w)
        predictors_t[0, l_title + index] = 1

    for t in x['tags']:
        index = all_tags.index(t)
        outputs.append(index)

        if nd == 0:
            predictors = predictors_t
        else:
            predictors = vstack([predictors, predictors_t])

    # If the index is evenly divisible by 500, print a message
    m.print_message(nd)

# Save the results to the file
dd.io.save(hfile, {
    'outputs': outputs,
    'predictors': predictors.tocsr(),
    'tags': all_tags,
    'title': all_title,
    'content': all_content
    })
