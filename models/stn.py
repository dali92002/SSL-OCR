import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 60, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )


    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * xs.size()[2] * xs.size()[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
