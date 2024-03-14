import torch
import torch.nn as nn

class convolution2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation):
        super(convolution2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.activation = nn.ReLU()
        self.act = activation

    def forward(self, x):
        out = self.conv(x)
        if self.act:
            out = self.activation(out)
        return out


class identity_Block(nn.Module):
    def __init__(self, in_channels, filter):
        super(identity_Block, self).__init__()

        self.branch1 = convolution2D(in_channels, filter, 3, True)

        self.activation = nn.ReLU()

    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = x

        out = torch.cat([branch1, branch2], 1)

        return self.activation(out)


class ResidualNet(nn.Module):
    def __init__(self, numChannels):
        super(ResidualNet, self).__init__()

        self.conv1 = convolution2D(in_channels=numChannels, out_channels=16, kernel_size=3, activation=True)
        self.MaxPooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = convolution2D(in_channels=16, out_channels=32, kernel_size=3, activation=True)
        self.IdentityBlock1 = identity_Block(32, 32)
        self.conv3 = convolution2D(in_channels=64, out_channels=80, kernel_size=3, activation=True)
        self.conv4 = convolution2D(in_channels=80, out_channels=100, kernel_size=3, activation=True)
        self.conv5 = convolution2D(in_channels=100, out_channels=120, kernel_size=3, activation=True)
        self.fc = nn.Linear(in_features=1505280 , out_features=8)
        self.activation = nn.LogSoftmax()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.MaxPooling1(out)

        out = self.conv2(out)

        out = self.IdentityBlock1(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        out = self.dropout(out)
        out = self.activation(out)

        return out