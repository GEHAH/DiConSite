import yaml
import random
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return argparse.Namespace(**{k: argparse.Namespace(**v) for k, v in config.items()})


def Seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def write_log(logfile,text,isPrint=True):
    if isPrint:
        print(text)
    
    if logfile != print:
        logfile.write(text)
        logfile.write('\n')



def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # euclidean_distance = F.pairwise_distance(output1, output2)
        cos_distance = D(output1, output2)
        # print("ED",euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 3))
        return loss_contrastive

def get_device():
    # if torch.cuda.is_available():
    #     print("Using CUDA GPU")
    #     return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     print("Using MPS (Apple Silicon)")
    #     return torch.device("mps")
    # else:
        print("Using CPU")
        return torch.device("cpu")
