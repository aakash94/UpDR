import torch
import torch.nn as nn
import torch.nn.functional as F

class UpNet(nn.Module):
    def __init__(self):
        super(UpNet, self).__init__()
        # convolutional layer
        self.up_layer = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=7, stride=1, padding=3)
        self.down_layer = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = F.elu(self.up_layer(x))
        x = F.elu(self.down_layer(x))        
        return x