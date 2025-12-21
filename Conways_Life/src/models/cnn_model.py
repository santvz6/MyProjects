import torch
import torch.nn as nn
import torch.nn.functional as F


class ConwayCNN(nn.Module):
    def __init__(self):
        super(ConwayCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2) # Reduces half the size
        self.dropout = nn.Dropout(0.25)
        
        # Originally: 28x28,
        # After: (x2 MaxPool2d) 7x7 and 64 channels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 labels

    def forward(self, x):
        # CNN
        x = self.pool(F.relu(self.conv1(x))) # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x))) # Conv -> ReLU -> Pool

        # MLP    
        x = x.view(-1, 64 * 7 * 7)  # Flatten (1, 64 x 7 x 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x