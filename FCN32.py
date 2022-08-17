import torch.nn as nn

class FCN32(nn.Module):
    def __init__(self,pretrained_net):
        super(FCN32, self).__init__()
        # 基础网络
        self.pretrained_net=pretrained_net
        # 全卷积
        self.model=nn.Sequential(

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self,x):
        x=self.pretrained_net(x)
        x=self.model(x)
        return x