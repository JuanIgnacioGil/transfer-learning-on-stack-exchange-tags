import csv
import os
from generate_wordlists import *

test_file = '../data/test.csv'
submission = '../data/submission.csv'

# Delete older submission file
try:
    os.remove(submission)
except OSError:
    pass


with open(submission, 'a') as s:
    writer = csv.writer(s, delimiter=',', lineterminator='\r\n', quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['id', 'tags'])

    with open(test_file, 'r') as t:
        test_reader = csv.reader(t, delimiter=',')
        next(test_reader, None)  # skip the headers

        for row in test_reader:

            row_id = row[0]
            title = titles_to_wordlist(row[1], remove_stopwords=True)
            content = content_to_wordlist(row[2], remove_stopwords=True)

            tags = 'physics poetry'
            print('{}: {}'.format(row_id, tags))
            writer.writerow([row_id, tags])

        t.close()

    s.close()
