import random
import numpy as np
import argparse
from model import *
from loaddatasets import *
from collections import Counter



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
    parser.add_argument('-beta', action='store', dest='beta', default=0.9, type=float)
    parser.add_argument('-eta', action='store', dest='eta', default=-0.01, type=float)
    parser.add_argument('-learningrate', action='store', dest='learningrate', default=0.01, type=float)
    parser.add_argument('-RecLossFunc', action='store', dest='RecLossFunc', default='Smooth')
    parser.add_argument('-DriftDetector', dest='DriftDetector', default='adwin',
                        choices=['adwin', 'mddm_g', 'mddm_a', 'mddm_e'])
    parser.add_argument('-MDDM_win',      dest='MDDM_win',      default=100,    type=int)
    parser.add_argument('-MDDM_ratio',    dest='MDDM_ratio',    default=1.01,   type=float)
    parser.add_argument('-MDDM_delta',    dest='MDDM_delta',    default=1e-6,   type=float)
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

        self.driftdet   = args.DriftDetector
        self.mddm_win   = args.MDDM_win
        self.mddm_ratio = args.MDDM_ratio
        self.mddm_delta = args.MDDM_delta

    def train(self):
        if self.datasetname == 'magic':
            print('magic trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadmagic()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 19019, 1919, 10, 30, 'parameter_magic', RecLossFunc=self.RecLossFunc)
            train.SecondPeriod()
            
        elif self.datasetname == 'adult':
            print('adult trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadadult()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 32559, 3559, 14, 30, 'parameter_adult', RecLossFunc=self.RecLossFunc)
            train.SecondPeriod()
            
        elif self.datasetname == 'car':
            print('car trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadcar()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 1380, 138, 6, 30, 'parameter_car', RecLossFunc=self.RecLossFunc)
            train.SecondPeriod()
        elif self.datasetname == 'arrhythmia':
            print('arrhythmia trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadarrhythmia()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 60, 10, 279, 30, path='parameter_arrhythmia', RecLossFunc=self.RecLossFunc)
            train.SecondPeriod()
        elif self.datasetname == 'thyroid':
            print('thyroid trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadthyroid()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 180, 18, 5, 30, 'parameter_thyroid', RecLossFunc=self.RecLossFunc)
            train.SecondPeriod()
        elif self.datasetname == 'insects':
            print('insects training starts')
            x_S1, y_S1, x_S2, y_S2 = loadinsects()

            out_path = 'parameter_insects' if self.driftdet == 'adwin' else 'parameter_insects_mddm'
            
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 
                                  2000, 500,
                                  dimension1=x_S1.shape[1],
                                  dimension2=30,
                                  path= out_path,
                                  RecLossFunc=self.RecLossFunc,
                                  lr = self.learningrate,
                                  b = self.beta,
                                  eta = self.eta,
                                  detector_type=self.driftdet,
                                  mddm_win=self.mddm_win,
                                  mddm_ratio=self.mddm_ratio,
                                  mddm_delta=self.mddm_delta)
            train.SecondPeriod()
        else:
            print('Choose a correct dataset name please')

if __name__ == '__main__':
    setup_seed(30)
    main()





