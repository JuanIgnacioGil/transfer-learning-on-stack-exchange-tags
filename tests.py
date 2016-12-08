import os
wdir = os.path.dirname(__file__)
from stack_exchange_tags.pos_analysis import POSAnalysis as PA


data_folder = os.path.join(wdir, 'data')
h5_file = os.path.join(wdir,'data/words_pandas.h5')
topics = ['biology', 'cooking', 'crypto', 'robotics', 'travel']
s = PA(h5_file, data_folder)
s.generate_hdf5_file(topics, 'data')
