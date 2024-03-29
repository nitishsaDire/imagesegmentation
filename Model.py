import torch
import torch.nn as nn
from torchvision import models

# created by Nitish Sandhu
# date 05/feb/2021

class UNET_resnet34(nn.Module):
    def __init__(self, n_classes, dropout_p=0.4):
        super().__init__()
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.resnet = models.resnet34(pretrained=True)
        self.resnet_layers = list(self.resnet.children())
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_7x7 = nn.Upsample((7,7), mode='nearest')


        self.seq1 = nn.Sequential(*self.resnet_layers[0:3])
        self.seq2 = nn.Sequential(*self.resnet_layers[3:5])
        self.seq3 = nn.Sequential(*self.resnet_layers[5])
        self.seq4 = nn.Sequential(*self.resnet_layers[6])
        self.seq5 = nn.Sequential(*self.resnet_layers[7])

        self.conv2d_0  = self.conv2d(512, 1024, kernel=3, stride = 2, padding=1)
        self.conv2d_01 = self.conv2d(1024, 2048, kernel=3, stride = 2, padding=1)
        self.adaptivePooling = nn.AdaptiveAvgPool2d((1,1))

        self.conv2d_00d = self.conv2d(2048 + 2048, 2048, kernel=3, padding=1)
        self.conv2d_01d = self.conv2d(2048 + 1024, 1024, kernel=3, padding=1)
        self.conv2d_02d = self.conv2d(1024 + 512, 512, kernel=3, padding=1)
        self.conv2d_1 = self.conv2d(512 + 256, 256, kernel=3, padding=1)
        self.conv2d_2 = self.conv2d(256 + 128, 128, kernel=3, padding=1)
        self.conv2d_3 = self.conv2d(128 + 64, 64, kernel=3, padding=1)
        self.conv2d_4 = self.conv2d(64 + 64, 64, kernel=3, padding=1)
        self.conv2d_f = self.conv2d_final(64, self.n_classes)

        self.softmax = nn.Softmax2d()

    def conv2d(self, in_channels, out_channels, kernel=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=self.dropout_p),
            nn.ReLU()
        ).to(self.device)

    def conv2d_final(self, in_channels, out_channels, kernel=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            # nn.BatchNorm2d(out_channels),
            # nn.Dropout2d(p=self.dropout_p),
            # nn.ReLU()
        ).to(self.device)

    def forward(self, x):
        """

        :param x: shape (bs, 3, 224, 224)
        :return:
        """
        x1e = self.seq1(x)          # 112x112, 64
        x2e = self.seq2(x1e)        # 56x56, 64
        x3e = self.seq3(x2e)        # 28x28, 128
        x4e = self.seq4(x3e)        # 14x14, 256
        x5e = self.seq5(x4e)        # 7x7, 512
        x6e = self.conv2d_0(x5e)    # 4x4, 1024
        x7e = self.conv2d_01(x6e)   # 2x2, 2048
        x8e = self.adaptivePooling(x7e)   # 1x1, 2048
        # print(x5e.shape)
        # print(x6e.shape)
        # print(x7e.shape)
        # print(x8e.shape)

        x7d = self.conv2d_00d(torch.cat((x7e, self.upsample(x8e)), 1))      # 2x2,      2048
        x6d = self.conv2d_01d(torch.cat((x6e, self.upsample(x7d)), 1))      # 3x3,      1024
        x5d = self.conv2d_02d(torch.cat((x5e, self.upsample_7x7(x6d)), 1))  # 7x7,      512
        x4d = self.conv2d_1(torch.cat((x4e, self.upsample(x5d)), 1))        # 14x14,    256
        x3d = self.conv2d_2(torch.cat((x3e, self.upsample(x4d)), 1))        # 28x28,    128
        x2d = self.conv2d_3(torch.cat((x2e, self.upsample(x3d)), 1))        # 56x56,    64
        x1d = self.conv2d_4(torch.cat((x1e, self.upsample(x2d)), 1))        # 112x112,  64

        final = self.conv2d_f(self.upsample(x1d))

        return final