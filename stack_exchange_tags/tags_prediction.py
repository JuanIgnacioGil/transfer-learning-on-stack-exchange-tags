import pandas as pd


class TagsPrediction:
    """
    Class to predict the set of tags of a given topic
    """

    def __init__(self, topic, file):

        self.topic = topic
        self.file = file

    def analize_tags(self):

        # Loads data
        store = pd.HDFStore(self.file)
        print(store[self.topic])
