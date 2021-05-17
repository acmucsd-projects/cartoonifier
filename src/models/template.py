import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Define model
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.first_conv_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.second_conv_layer = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(img_shape) * 8, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.first_conv_layer(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.second_conv_layer(x))
        x = F.max_pool2d(x, kernel_size=2)
        return self.linear_relu_stack(x)