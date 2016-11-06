import json
import pickle
import numpy as np
import time

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
predictors = []
outputs = []

for x, nd in zip(data, range(l_data)):

    # Generate output vector
    x_outputs = np.zeros(l_tags)

    for t, n in zip(all_tags, range(l_tags)):
        if t in x['tags']:
            x_outputs[n] = 1

    # Generate predictor vector

    x_predictors = np.zeros(l_title + l_content)

    for w, n in zip(all_title, range(l_title)):
        if w in x['title']:
            x_predictors[n] = 1

    for w, n in zip(all_content, range(l_content)):
        if w in x['content']:
            x_predictors[l_title + n] = 1

    predictors.append(x_predictors)
    outputs.append(x_outputs)

    # If the index is evenly divisible by 100, print a message
    if (nd + 1) % 100 == 0:
        p = int((100 * (nd + 1) / l_data))
        elapsed = time.time() - t0
        remaining = int(elapsed * (l_data - nd - 1) / (60 * (nd + 1)))
        print('{}% calculated. {} minutes remaining'.format(p, remaining))


# Save arrays to picke file
with open('../data/model_data.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([predictors, outputs], f)

# Save list of unique tags and words
unique_words = {
        'tags': list(all_tags),
        'title': list(all_title),
        'content': list(all_content)
    }

with open('../data/unique_words.json', 'w') as outfile:
    json.dump(unique_words, outfile, sort_keys=False, indent=4)
