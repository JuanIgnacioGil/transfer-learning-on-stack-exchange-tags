import os
wdir = os.path.dirname(__file__)
from stack_exchange_tags.validate_and_test import StackExchangeTags as stex

s = stex(
    train_file=os.path.join(wdir, 'data/data.h5'),
    validation_file=os.path.join(wdir, 'data/naive_bayes_validation_output.txt'),
    test_file=os.path.join(wdir, 'data/test.h5'),
    test_json_file=os.path.join(wdir, 'data/test.json'),
    test_csv_file=os.path.join(wdir, 'data/test.csv'),
    submission=os.path.join(wdir, 'data/submission.csv'),
    n=2,
)

#s.generate_predictors_test()

#Validate
s.validate()

#Test
#s.generate_submission()

