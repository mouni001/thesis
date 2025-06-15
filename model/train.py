import random
import numpy as np
import argparse
from model import *
from loaddatasets import *

# During the overlap period (T1):

# You have access to both x_S1 and x_S2

# You train two VAEs:

# VAE1 encodes x_S1 → z_S1

# VAE2 encodes x_S2 → z_S2

# Then you:

# Force z_S1 ≈ z_S2 (using KL divergence)

# Train a decoder to go from z_S2 → reconstruct x_S1
# (so even when S1 disappears, you can simulate it)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-DataName', action='store', dest='DataName', default='enfr')
    parser.add_argument('-AutoEncoder', action='store', dest='AutoEncoder', default='VAE')
    parser.add_argument('-beta', action='store', dest='beta', default=0.9)
    parser.add_argument('-eta', action='store', dest='eta', default=-0.01)
    parser.add_argument('-learningrate', action='store', dest='learningrate', default=0.01)
    parser.add_argument('-RecLossFunc', action='store', dest='RecLossFunc', default='Smooth')
    args = parser.parse_args()
    learner = OLD3S(args)
    learner.train()


class OLD3S:
    def __init__(self, args):
        '''
            Data is stored as list of dictionaries.
            Label is stored as list of scalars.
        '''
        self.datasetname = args.DataName
        self.autoencoder = args.AutoEncoder
        self.beta = args.beta
        self.eta = args.eta
        self.learningrate = args.learningrate
        self.RecLossFunc = args.RecLossFunc

    def train(self):
        if self.datasetname == 'magic':
            print('magic trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadmagic()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 19019, 1919, 10, 30, 'parameter_magic')
            train.SecondPeriod()
            torch.save(train.result['acc'], './data/parameter_magic/Accuracy')
            x_vals = np.array([i for i in range(500, 500 * (len(train.result['acc']) + 1), 500)])
            plot_reuter(np.array(train.result['acc']), x_vals,
                        './data/parameter_magic/Accuracy_Curve.png',
                        45000, 50000)
        elif self.datasetname == 'adult':
            print('adult trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadadult()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 32559, 3559, 14, 30, 'parameter_adult')
            train.SecondPeriod()
            torch.save(train.result['acc'], './data/parameter_magic/Accuracy')
            x_vals = np.array([i for i in range(500, 500 * (len(train.result['acc']) + 1), 500)])
            plot_reuter(np.array(train.result['acc']), x_vals,
                        './data/parameter_magic/Accuracy_Curve.png',
                        45000, 50000)
        else:
            print('Choose a correct dataset name please')

if __name__ == '__main__':
    setup_seed(30)
    main()





