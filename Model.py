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
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.softmax = nn.Softmax(dim=1)
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    def conv2d(self, in_channels, out_channels, kernel = 1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding = padding),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout2d(p=self.dropout_p),
            nn.ReLU()).to(self.device)

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


        x4d = self.conv2d(512 + 256, 256, kernel=3, padding=1)(torch.cat((x4e, self.upsample(x5e)), 1))            #14x14,    256
        x3d = self.conv2d(256 + 128, 128, kernel=3, padding=1)(torch.cat((x3e, self.upsample(x4d)), 1))            #28x28,    128
        x2d = self.conv2d(128 + 64, 64, kernel=3, padding=1)(torch.cat((x2e, self.upsample(x3d)), 1))             #56x56,    64
        x1d = self.conv2d(64+64, 64, kernel=3, padding=1)(torch.cat((x1e, self.upsample(x2d)), 1))                   #112x112,  64

        final = self.conv2d(64, self.n_classes)(self.upsample(x1d))

        return self.softmax(final)