import os
wdir = os.path.dirname(__file__)
from stack_exchange_tags.word_frequencies import WordFrequencies as WF

s = WF(
    hfile = os.path.join(wdir,'data/frequencies.h5'),
    json_data = os.path.join(wdir,'data/data.json'),
    submission = os.path.join(wdir,'data/submission.csv'),
    validation_file = os.path.join(wdir, 'data/frequencies_validation_output.csv'),
    test_hfile = os.path.join(wdir, 'data/test.h5'),
    test_json = os.path.join(wdir, 'data/test.json')
)

#s.generate_predictors_test()
s.generate_word_test()

#Validate
#s.validate()

#Test
#s.generate_submission()