import os
dir = os.path.dirname(__file__)
from stack_exchange_tags.validate_and_test import StackExchangeTags as stex

s = stex(
    train_file=os.path.join(dir,'data/data.h5'),
    validation_file=os.path.join(dir,'data/naive_bayes_validation_output.txt'),
    test_file=os.path.join(dir,'data/test.h5'),
    test_json_file=os.path.join(dir,'data/test.json'),
    test_csv_file=os.path.join(dir,'data/test.csv'),
    submission=os.path.join(dir,'data/submission.csv'),
)

#Validate
s.validate()

#Test
#s.generate_submission()

