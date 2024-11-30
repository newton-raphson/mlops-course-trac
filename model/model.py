import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(LogisticRegression, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        return self.hidden_layers(x)