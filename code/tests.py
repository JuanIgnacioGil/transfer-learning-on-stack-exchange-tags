from stack_exchange_tags import StackExchangeTags as stex
import numpy as np

train_file = '../data/data.h5'
output_file = '../data/naive_bayes_validation_output.txt'
test_file = '../data/test.h5'
test_json_file = '..data/test.json'
test_csv_file = '..data/test.csv'
submission = '../data/submission.csv'

test = 'NaiveBayesTest'

if test == 'NaiveBayesValidation':

    x_train, x_validation, y_train, y_validation = stex.validation_sets(train_file)
    y_predict = stex.naive_bayes(x_train, y_train, x_validation)
    stex.validate(y_predict, y_validation, train_file, output_file)

elif test == 'NaiveBayesTest':

    x_train, y_train, x_test = stex.tests_sets(train_file, test_file)


    x_groups = np.array_split(x_test, 10)
    y_predict = np.empty((1, y_train.shape[1]))

    for group in x_groups:
            y_predict = np.concatenate(y_predict, stex.naive_bayes(x_train, y_train, group))

    stex.generate_submission(y_predict, train_file, test_json_file, submission)