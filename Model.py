import torch
import torch.nn as nn
from torchvision import models

# created by Nitish Sandhu
# date 05/feb/2021

class UNET_resnet34(nn.Module):
    def __init__(self, n_classes, dropout_p = 0.4):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.resnet = models.resnet34(pretrained=True)
        self.resnet_layers = list(self.resnet.children())
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    def conv2d(self, in_channels, out_channels, kernel = 1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=self.dropout_p),
            nn.ReLU())

    def forward(self, x):
        """

        :param x: shape (bs, 3, 224, 224)
        :return:
        """
        x1e = nn.Sequential(*self.resnet_layers[0:3])(x)              #112x112, 64
        x2e = nn.Sequential(*self.resnet_layers[3:5])(x1e)            #56x56, 64
        x3e = nn.Sequential(*self.resnet_layers[5])(x2e)              #28x28, 128
        x4e = nn.Sequential(*self.resnet_layers[6])(x3e)              #14x14, 256
        x5e = nn.Sequential(*self.resnet_layers[7])(x4e)              #7x7, 512

        # print(x4e.shape)
        # print(x5e.shape)
        # print(torch.cat((x4e, self.conv2d(512, 256)(self.upsample(x5e))), 1).shape)
        # print(self.conv2d(512, 256)(self.upsample(x5e)).shape)
        conv2dd = self.conv2d(512, 256)(self.upsample(x5e))
        catt = torch.cat((x4e, conv2dd), 1)
        x4d = self.conv2d(512, 256, kernel=3, padding=1)(catt)            #14x14,    256
        x3d = self.conv2d(256, 128, kernel=3, padding=1)(torch.cat((x3e, self.conv2d(256, 128)(self.upsample(x4d))), 1))            #28x28,    128
        x2d = self.conv2d(128, 64, kernel=3, padding=1)(torch.cat((x2e, self.conv2d(128,  64)(self.upsample(x3d))), 1))              #56x56,    64
        x1d = self.conv2d(128, 64, kernel=3, padding=1)(torch.cat((x1e, self.upsample(x2d)), 1))                                     #112x112,  64

        final = self.conv2d(64, self.n_classes)(self.upsample(x1d))

        return self.sigmoid(final)