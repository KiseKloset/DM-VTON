import time
import torch
import torch.nn as nn


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_channels):
        super(MobileNet, self).__init__()
        self.layer1 = nn.Sequential(
            conv_bn(n_channels, 32, 1),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )

        self.layer2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )

        self.layer3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

    def forward(self, x):
        out0 = self.layer1(x)
        out1 = self.layer2(out0)
        out2 = self.layer3(out1)
        return out0, out1, out2


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
    
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class MobileNetUNet(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(MobileNetUNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes

        self.backbone = MobileNet(n_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = DoubleConv(1024, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv3 = DoubleConv(512, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        P5 = self.up1(x0)
        P5 = self.conv1(P5)           

        P4 = x1                       
        P4 = torch.cat([P4, P5], axis=1)  

        P4 = self.up2(P4)            
        P4 = self.conv2(P4)          
        P3 = x2                      
        P3 = torch.cat([P4, P3], axis=1)

        P3 = self.up3(P3)
        P3 = self.conv3(P3)

        P3 = self.up4(P3)
        P3 = self.conv4(P3)

        out = self.out(P3)
        return out

if __name__ == "__main__":
    device = torch.device("cuda:1")
    net = MobileNetUNet(7, 4).to(device)
    net.eval()
    x = torch.rand(1, 7, 256, 192).to(device)
    with torch.no_grad():
        out = net(x)
        start = time.time()
        for i in range(1000):
            out = net(x)
        end = time.time()
    print(out.shape)
    print((end - start) / 1000)
