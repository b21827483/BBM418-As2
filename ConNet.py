from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import Dropout

from torch import flatten

class ConNet(Module):
    def __init__(self, numChannels):
        super(ConNet, self).__init__()


        self.conv1 = Conv2d(in_channels=numChannels, out_channels=16, kernel_size=(3, 3))
        self.relu1 = ReLU()
        self.maxpool = MaxPool2d(kernel_size=(2, 2),stride=(2, 2))

        self.conv2 = Conv2d(16, 32, kernel_size=(3, 3))
        self.relu2 = ReLU()

        self.conv3 = Conv2d(32, 40, kernel_size=(3, 3))
        self.relu3 = ReLU()

        self.conv4 = Conv2d(40, 50, kernel_size=(3, 3))
        self.relu4 = ReLU()

        self.conv5 = Conv2d(50, 60, kernel_size=(3, 3))
        self.relu5 = ReLU()

        self.conv6 = Conv2d(60, 70, kernel_size=(3, 3))
        self.relu6 = ReLU()

        self.fc1 = Linear(in_features=714070, out_features=8)

        self.logSoftmax = LogSoftmax(dim=1)

        self.dropout = Dropout(0.5)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)

        x = self.dropout(x)

        output = self.logSoftmax(x)
        return output