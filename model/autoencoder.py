import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder_Shallow(nn.Module):
    """ AutoEncoder for shallow data (e.g., tabular data)
        Purely fully connected layers, no convolutions
        Takes vectors as input, outputs reconstructed vectors
        Encoder maps the feature size from inplanes to outplanes
        Decoder maps the feature size from outplanes to inplanes
        Encoder and decoder are symmetric, but not identical
        Encoder uses ReLU, decoder uses LeakyReLU
        Encoder uses BatchNorm, decoder uses InstanceNorm
        Encoder uses dropout, decoder uses dropout
        Encoder uses a final linear layer to map the feature size from inplanes to outplanes
        Decoder uses a final linear layer to map the feature size from outplanes to inplanes
        """

    def __init__(self, inplanes, outplanes):
        super(AutoEncoder_Shallow, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inplanes, outplanes),
            nn.ReLU()

        )
        self.decoder = nn.Sequential(
            nn.Linear(outplanes, inplanes),
            nn.ReLU()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder






