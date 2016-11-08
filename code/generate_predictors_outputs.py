import json
import time
import deepdish as dd
import scipy.sparse as sp

hfile='../data/data.h5'

# Read the json file
with open('../data/data.json') as json_data:
    data = json.load(json_data)
    json_data.close()

l_data = len(data)

# Generate a list of all possible tags
all_tags_list = []

for x in data:
    all_tags_list += x['tags']


all_tags = set(all_tags_list)
l_tags = len(all_tags)

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
outputs = sp.lil_matrix((l_data, l_tags))

for x, nd in zip(data, range(l_data)):
#for x, nd in zip(data[:100], range(100)):

    for t, n in zip(all_tags, range(l_tags)):
        if t in x['tags']:
            outputs[nd, n] = 1

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


# Save list of unique tags and words
unique_words = {
        'tags': list(all_tags),
        'title': list(all_title),
        'content': list(all_content)
    }


dd.io.save(hfile, {
    'outputs': outputs.tocsr(),
    'predictors': predictors.tocsr(),
    'tags': list(all_tags),
    'title': list(all_title),
    'content': list(all_content)
    })
