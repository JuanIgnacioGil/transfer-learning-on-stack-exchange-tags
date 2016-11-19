from stack_exchange_tags import StackExchangeTags as stex

train_file = '../data/data.h5'
output_file = '../data/naive_bayes_validation_output.txt'

x_train, x_validation, y_train, y_validation = stex.validation_sets(train_file)
y_predict = stex.naive_bayes(x_train, y_train, x_validation)
stex.validate(y_predict, y_validation, train_file, output_file)
