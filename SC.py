import numpy as np
import torch
import torch.nn as nn
from torch import Tensor



from High_Frequency_Module import HighFrequencyModule



class SC(nn.Module):
    def __init__(self, in_channels, r=0.5):
        super(SC, self).__init__()
        self.in_channel = in_channels

        # ***************************第一层********************************************

        # 平均池化
        self.av1 = nn.AdaptiveAvgPool2d(1)

        # 全连接层y=wx+b
        self.lin1 = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * r)),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(int(in_channels * r), in_channels),
            nn.Sigmoid(),
        )

        # 3*3卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=2),
            nn.BatchNorm2d(in_channels, 0.1),
            nn.ReLU(in_channels)
        )

        # ***************************第二层********************************************

        # 平均池化
        self.av2 = nn.AdaptiveAvgPool2d(1)

        # 全连接层y=wx+b
        self.lin2 = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * r)),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(int(in_channels * r), in_channels),

            nn.Sigmoid(),
        )

        # 3*3卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels= in_channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(in_channels, 0.1),
            nn.ReLU(in_channels)
        )
        self.conv21 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 8, kernel_size=3, padding=1),

            nn.BatchNorm2d(in_channels * 8, 0.1),
            nn.ReLU(in_channels * 8)
        )
        # ***************************第三层********************************************

        # 平均池化
        self.av3 = nn.AdaptiveAvgPool2d(1)

        # 全连接层y=wx+b
        self.lin3 = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * r)),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(int(in_channels * r), in_channels),
            nn.Sigmoid(),
        )

        # 3*3卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=2),
            nn.BatchNorm2d(in_channels, 0.1),
            nn.ReLU(in_channels)
        )
        self.conv31 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 8, kernel_size=3, padding=1),

            nn.BatchNorm2d(in_channels * 8, 0.1),
            nn.ReLU(in_channels * 8)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 16, out_channels=in_channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(in_channels , 0.1),
            nn.ReLU(in_channels )
        )

        self.mp = nn.MaxPool2d(8)

        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))
        self.w3 = nn.Parameter(torch.ones(2))

        self.SA = nn.Softmax(dim=-1)

    def forward(self, x):

        # 对x进行分支计算权重, 进行全局均值池化
        branch1 = self.av1(x)
        branch1 = branch1.view(branch1.size(0), -1)

        # 全连接层得到权重
        weight1 = self.lin1(branch1)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight1.shape
        weight1 = torch.reshape(weight1, (h, w, 1, 1))

        # 乘积获得结果
        x11 = weight1 * x


        x12 = self.conv1(x)

        # 归一化权重
        w11 = torch.exp(self.w1[0]) / torch.sum(torch.exp(self.w1))
        w12 = torch.exp(self.w1[1]) / torch.sum(torch.exp(self.w1))

        x1 = x11 * w11 + x12 * w12


        #---------------------------1 END---------------------------------

        # 对x进行分支计算权重, 进行全局均值池化
        branch2 = self.av2(x)
        branch2 = branch2.view(branch2.size(0), -1)

        # 全连接层得到权重
        weight2 = self.lin2(branch2)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        hh, ww = weight2.shape
        weight2 = torch.reshape(weight2, (hh, ww, 1, 1))

        # 乘积获得结果
        x21 = weight2 * x

        x22 = self.conv2(x)

        # 归一化权重
        w21 = torch.exp(self.w2[0]) / torch.sum(torch.exp(self.w2))
        w22 = torch.exp(self.w2[1]) / torch.sum(torch.exp(self.w2))

        x2 = x21 * w21 + x22 * w22
        x2 = self.conv21(x2)

        #------------------------------2 END------------------------------------

        # 对x进行分支计算权重, 进行全局均值池化
        branch3 = self.av3(x)
        branch3 = branch3.view(branch3.size(0), -1)

        # 全连接层得到权重
        weight3 = self.lin3(branch3)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        hhh, www = weight3.shape
        weight3 = torch.reshape(weight3, (hhh, www, 1, 1))

        # 乘积获得结果
        x31 = weight3 * x

        x32 = self.conv3(x)

        # 归一化权重
        w31 = torch.exp(self.w3[0]) / torch.sum(torch.exp(self.w3))
        w32 = torch.exp(self.w3[1]) / torch.sum(torch.exp(self.w3))

        x3 = x31 * w31 + x32 * w32

        x3 = self.conv31(x3)

        #---------------------------------3 END-----------------------------------
        x1 = torch.transpose(x1, 0, 3)
        x1 = torch.transpose(x1, 1, 2)
        x2 = torch.transpose(x2, 0, 2)
        x2 = torch.transpose(x2, 1, 3)

        xx = torch.matmul(x1, x2)
        xx = self.SA(xx)

        x3 = torch.transpose(x3, 0, 3)
        x3 = torch.transpose(x3, 1, 2)
        x = torch.matmul(xx, x3)
        x = torch.transpose(x, 0, 3)
        x = torch.transpose(x, 1, 2)

        x = self.SA(x)








        return x








