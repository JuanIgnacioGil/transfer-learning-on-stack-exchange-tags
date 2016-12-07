import os
wdir = os.path.dirname(__file__)
from stack_exchange_tags.pos_analysis import POSAnalysis as PA

s = PA(os.path.join(wdir,'data/pos_freq.h5'))
s.generate_hdf5_file('biology', os.path.join(wdir,'data/biology.csv'))
