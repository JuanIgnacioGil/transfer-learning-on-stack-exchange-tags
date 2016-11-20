from time import time

class IterMessage():

    def __init__(self, total_items, activity, step):

        self.total_items = total_items
        self.activity = activity
        self.step = step
        self.t0 = time()

    def print_message(self, item):
        # If the index is evenly divisible by 200, print a message
        if (item + 1) % self.step == 0:
            p = int((100 * (item + 1) / self.total_items))
            elapsed = time() - self.t0
            remaining = int(elapsed * (self.total_items - item - 1) / (60 * (item + 1)))
            print('{}% {}. {} minutes remaining'.format(p, self.activity, remaining))