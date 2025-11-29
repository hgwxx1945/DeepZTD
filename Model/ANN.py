import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, input_size=5,layernumber=4, hidden_size=256,output_size=1):
        super(ANN, self).__init__()
        self.hidden_size = hidden_size
        self.layernumber = layernumber
        # using for loop to create layers
        self.input = nn.Linear(input_size, hidden_size)
        for i in range(layernumber):
            setattr(self, 'fc{}'.format(i), nn.Linear(hidden_size, hidden_size))
        self.output = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = F.relu(self.input(x))
        for i in range(self.layernumber):
            x = F.relu(getattr(self, 'fc{}'.format(i))(x))
        x = self.output(x)
        return  x