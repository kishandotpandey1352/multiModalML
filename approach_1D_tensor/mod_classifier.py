import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralModalityEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], num_classes=4):
        super(GeneralModalityEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.output = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)
