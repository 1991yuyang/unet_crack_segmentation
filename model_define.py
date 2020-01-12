from torch import nn
import torch as t


class Conv3X3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv3X3, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class DeConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DeConv, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Conv1X1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv1X1, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.block(x)


class Unet(nn.Module):

    def __init__(self, in_channels, num_class):
        super(Unet, self).__init__()
        down_path = []
        up_path = []
        for i in range(5):
            if i == 0:
                down_path.append(nn.Sequential(
                    Conv3X3(in_channels=in_channels, out_channels=2 ** (6 + i)),
                    Conv3X3(in_channels=2 ** (6 + i), out_channels=2 ** (6 + i))
                ))
            else:
                down_path.append(nn.Sequential(
                    Conv3X3(in_channels=2 ** (5 + i), out_channels=2 ** (6 + i)),
                    Conv3X3(in_channels=2 ** (6 + i), out_channels=2 ** (6 + i))
                ))
            if i != 4:
                down_path.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if i <= 3:
                up_path.append(DeConv(in_channels=2 ** (10 - i), out_channels=2 ** (9 - i)))
                up_path.append(nn.Sequential(
                    Conv3X3(in_channels=2 ** (10 - i), out_channels=2 ** (9 - i)),
                    Conv3X3(in_channels=2 ** (9 - i), out_channels=2 ** (9 - i))
                ))
                if i == 3:
                    up_path.append(Conv1X1(in_channels=2 ** (9 - i), out_channels=num_class))
        self.down_path = nn.Sequential(*down_path)
        self.up_path = nn.Sequential(*up_path)

    def forward(self, x):
        fine_grained_features = []
        for n, m in self.down_path._modules.items():
            x = m(x)
            if int(n) % 2 == 0 and int(n) != 8:
                fine_grained_features.append(x)
        for n, m in self.up_path._modules.items():
            if int(n) % 2 != 0:
                fine_grained_feature = fine_grained_features.pop()
                crop_size = int((fine_grained_feature.size()[3] - x.size()[3]) / 2)
                concatenate_feature = fine_grained_feature[:, :, crop_size:-crop_size, crop_size:-crop_size]
                x = t.cat((concatenate_feature, x), dim=1)
            x = m(x)
        return x

