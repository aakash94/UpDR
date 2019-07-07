import torch
import torch.nn as nn
import torch.nn.functional as F

class UpNet(nn.Module):
    def __init__(self):
        super(UpNet, self).__init__()
        self.up_layer = nn.Conv2d(in_channels=3, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.mid_layer = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.down_layer = nn.Conv2d(in_channels=1024, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.elu(self.up_layer(x))
        x = F.elu(self.mid_layer(x))
        x = F.sigmoid(self.down_layer(x))        
        return x