from dependencies.spiralnet import instantiate_spiralnet
import torch
from torch import nn
import torch.nn.functional as F

class DummyAudioEncoder(nn.Module):
    def __init__(self):
        super(DummyAudioEncoder, self).__init__()
        self.ff1 = nn.Linear(2048, 1024)
        self.ff2 = nn.Linear(1024, 512)
        self.ff3 = nn.Linear(512, 256)
        self.ff4 = nn.Linear(256, 128)
    
    def forward(self, x):
        x = F.relu_(self.ff1(x))
        x = F.relu_(self.ff2(x))
        x = F.relu_(self.ff3(x))
        x = F.relu_(self.ff4(x))
        return x