import torch
import torch.nn as nn
from torch.nn import functional as F, Softmax

from ECA import ECABlock
from diversebranchblock import DiverseBranchBlock

class VGGBlock(nn.Module):
    def __init__(self, in_channels,  out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels=out_channels // 2,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels //2 )
        self.conv2 = nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class ResidualBlock(nn.Module):#ResidualBlock
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()#初始化层必须要加这个是吗
        #self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels= out_channels // 2, kernel_size = 3, padding = 1)#先做卷积
        self.conv2 = nn.Conv2d( in_channels=out_channels // 2, out_channels=out_channels, kernel_size =3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.relu(self.conv1(x))#然后激活
        y = self.conv2(y)
        x = self.conv3(x)
        return self.relu(x+y)

def call(x1,x2):
    #assert isinstance(x, list)
    x1 = torch.relu((x1 + x2) / 2)
    x1 = torch.sigmoid(x1)
    x2 = torch.relu(x2)
    M1 = x1
    M2 = x2
    YT = torch.mean(M2)
    Trp = torch.max(M1) / torch.min(M1)  # (torch.max(M1) + torch.mean(M1)) / (torch.mean(M1) - torch.min(M1))
    volumes = torch.multiply(Trp, M1)  #
    spectral = torch.where(M1 > YT, volumes, M2)  # 清零后被替换torch.where(M1 > M2, reward, M2)  # 0.4
    reward = 2.5 * M1
    punishment = torch.zeros_like(spectral)  # tf.zeros_like(M1)
    M1 = torch.where(M1 > 0.4, reward, punishment)  # tf.where(M1 > 0.2, x=reward, y=punishment)
    A = torch.multiply(M1, M2)
    return A


class Visual_Enhance(nn.Module):
    def __init__(self, in_channels):
        super(Visual_Enhance, self).__init__()
        self.in_channel = in_channels


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 16, kernel_size=1, stride=2),
            nn.BatchNorm2d(in_channels // 16, momentum=0.1),
            nn.ReLU(in_channels // 16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, stride=2),
            nn.BatchNorm2d(in_channels // 4, 0.1),
            nn.PReLU(in_channels // 4)
        )
        self.conv3 = nn.Sequential(
           # DiverseBranchBlock(in_channels=in_channels // 4, out_channels=in_channels // 8, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(in_channels // 8, momentum=0.1),
            nn.PReLU(in_channels // 8)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 8, out_channels=in_channels // 16, kernel_size=3, stride=2),
            #DiverseBranchBlock(in_channels=in_channels // 8, out_channels=in_channels // 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 16, momentum=0.1),
            nn.PReLU(in_channels // 16)
        )

        self.up = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        )
        self.rblock1 = ResidualBlock(in_channels= in_channels, out_channels=in_channels)

        self.mp = nn.MaxPool2d(4)

        self.softmax = Softmax(dim=-1)
        self.sk = ECABlock(channels=in_channels)

        self.Sigmoid = nn.Sigmoid()

        self.vgg1 = VGGBlock(in_channels=in_channels, out_channels=in_channels // 16)

        self.vgg2 = VGGBlock(in_channels=in_channels,  out_channels=in_channels // 4)

        self.vgg3 = VGGBlock(in_channels=in_channels // 4, out_channels=in_channels // 8)

        self.vgg4 = VGGBlock(in_channels=in_channels // 8,  out_channels=in_channels // 4)
        self.rblock1 = ResidualBlock(in_channels=in_channels // 4  , out_channels=in_channels)

    def forward(self, x):
        x1 = x
        c2 = self.vgg2(x)
        c2 = self.vgg3(c2)
        c2 = self.vgg4(c2)



        c2 = self.rblock1(c2)

        # x = torch.mul(x1, c2)

        # x = self.Sigmoid(x)
        x2 = c2
        x1 = torch.relu((x1 + x2) / 2)
        x1 = torch.sigmoid(x1)
        x2 = torch.relu(x2)
        M1 = x1
        M2 = x2
        YT = torch.mean(M2)
        Trp = torch.max(M1) / torch.min(M1)  # (torch.max(M1) + torch.mean(M1)) / (torch.mean(M1) - torch.min(M1))
        volumes = torch.multiply(Trp, M1)  #
        spectral = torch.where(M1 > YT, volumes, M2)  # 清零后被替换torch.where(M1 > M2, reward, M2)  # 0.4
        reward = 2.5 * M1
        punishment = torch.zeros_like(spectral)  # tf.zeros_like(M1)
        M1 = torch.where(M1 > 0.4, reward, punishment)  # tf.where(M1 > 0.2, x=reward, y=punishment)
        A = torch.multiply(M1, M2)
        return A


