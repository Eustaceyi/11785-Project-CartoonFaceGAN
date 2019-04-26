import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--")