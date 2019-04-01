import argparse

class Options():
    def __init__(self, mode):
        self.mode = mode
        if mode == 'train':
            return self.train_opts()
        elif mode == 'test':
            return self.test_opts()
        else:
            raise NotImplementedError('Mode [%s] is not found' % mode)

    def base_opts(self):
        

    def train_opts(self):

    def test_opts(self):
