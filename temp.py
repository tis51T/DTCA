import torch
from torchcrf import CRF
num_tags = 5  # number of tags is 5
model = CRF(num_tags)