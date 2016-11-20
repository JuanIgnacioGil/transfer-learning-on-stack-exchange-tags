from stack_exchange_tags import StackExchangeTags as stex

s = stex(
    train_file='data/data.h5',
    validation_file='data/naive_bayes_validation_output.txt',
    test_file='data/test.h5',
    test_json_file='data/test.json',
    test_csv_file='data/test.csv',
    submission='data/submission.csv',
)

