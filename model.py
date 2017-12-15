from collections import OrderedDict

from torch import nn
from torch.nn import init

from lsoftmax import LSoftmaxLinear


class MNISTModel(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

        cnn_layers = OrderedDict()
        cnn_layers['conv0_0'] = nn.Conv2d(in_channels=1, out_channels=64,
                                          kernel_size=(3, 3), padding=1)
        cnn_layers['prelu0_0'] = nn.PReLU(64)
        cnn_layers['bn0_0'] = nn.BatchNorm2d(64)
        # conv1.x
        for x in range(3):
            cnn_layers[f'conv1_{x}'] = nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3),
                padding=1)
            cnn_layers[f'prelu1_{x}'] = nn.PReLU(64)
            cnn_layers[f'bn1_{x}'] = nn.BatchNorm2d(64)
        cnn_layers['pool1'] = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # conv2.x
        for x in range(4):
            cnn_layers[f'conv2_{x}'] = nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3),
                padding=1)
            cnn_layers[f'prelu2_{x}'] = nn.PReLU(64)
            cnn_layers[f'bn2_{x}'] = nn.BatchNorm2d(64)
        cnn_layers['pool2'] = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # conv3.x
        for x in range(4):
            cnn_layers[f'conv3_{x}'] = nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3),
                padding=1)
            cnn_layers[f'prelu3_{x}'] = nn.PReLU(64)
            cnn_layers[f'bn3_{x}'] = nn.BatchNorm2d(64)
        cnn_layers['pool3'] = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.net = nn.Sequential(cnn_layers)
        self.fc = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=576, out_features=256)),
            # ('fc1', nn.Linear(in_features=256, out_features=10))
            ('fc0_bn', nn.BatchNorm1d(256))
        ]))

        self.lsoftmax_linear = LSoftmaxLinear(
            input_dim=256, output_dim=10, margin=margin)
        self.reset_parameters()

    def reset_parameters(self):
        def init_kaiming(layer):
            init.kaiming_normal(layer.weight.data)
            init.constant(layer.bias.data, val=0)

        init_kaiming(self.net.conv0_0)
        for x in range(3):
            init_kaiming(getattr(self.net, f'conv1_{x}'))
        for x in range(4):
            init_kaiming(getattr(self.net, f'conv2_{x}'))
        for x in range(4):
            init_kaiming(getattr(self.net, f'conv3_{x}'))
        init_kaiming(self.fc.fc0)
        self.lsoftmax_linear.reset_parameters()
        # init_kaiming(self.fc.fc1)

    def forward(self, input, target=None):
        """
        Args:
            input: A variable of size (N, 1, 28, 28).
            target: A long variable of size (N,).

        Returns:
            logit: A variable of size (N, 10).
        """

        conv_output = self.net(input)
        batch_size = conv_output.size(0)
        fc_input = conv_output.view(batch_size, -1)
        fc_output = self.fc(fc_input)
        logit = self.lsoftmax_linear(input=fc_output, target=target)
        return logit
