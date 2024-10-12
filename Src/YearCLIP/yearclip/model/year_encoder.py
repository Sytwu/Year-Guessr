import numpy as np
import torch
import torch.nn as nn
from .rff import GaussianEncoding
from .misc import file_dir

def normalize(years):
    return (years - 1000) / (2024 - 1000)

def equal_frequency_binning(years):
    # bin_edges is computed by stat.ipynb
    
    bin_edges = np.load('bin_edges.npy')
    num_bins = len(bin_edges) - 1
    
    new_years = np.digitize(years, bin_edges, right=True)
    new_years = (new_years - 1) / (num_bins - 1)
    
    return new_years

class YearEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(YearEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=1, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

class YearEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], from_pretrained=True):
        super(YearEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('YearEnc' + str(i), YearEncoderCapsule(sigma=s))

        if from_pretrained:
            self._load_weights()

    def _load_weights(self):
        self.load_state_dict(torch.load(f"{file_dir}/weights/year_encoder_weights.pth"))

    def forward(self, years):
        years = normalize(years)
        # years = equal_frequency_binning(years)
        years_features = torch.zeros(years.shape[0], 512).to(years.device)

        for i in range(self.n):
            years_features += self._modules['YearEnc' + str(i)](years)
        
        return years_features