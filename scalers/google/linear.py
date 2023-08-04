import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(8192, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 128)  # Second fully connected layer
        self.fc3 = nn.Linear(128, 1)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc2(x)
        x = nn.functional.leaky_relu(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc3(x)
        return x

# model = LinearModel()
# a = torch.rand(1, 8192)
# res = model(a)
# print(res)
