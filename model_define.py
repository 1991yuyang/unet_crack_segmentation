from torch import nn
import torch as t
from torch.nn import functional as F


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

    def __init__(self, in_channels, num_class, is_attention):
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
        self.is_attention = is_attention
        if is_attention:
            self.att1 = AttentionBlock(1024, 512, 64)
            self.att2 = AttentionBlock(512, 256, 64)
            self.att3 = AttentionBlock(256, 128, 64)
            self.att4 = AttentionBlock(128, 64, 64)
            self.attention_lst = [self.att1, self.att2, self.att3, self.att4]

    def forward(self, x):
        fine_grained_features = []
        for n, m in self.down_path._modules.items():
            x = m(x)
            if int(n) % 2 == 0 and int(n) != 8:
                fine_grained_features.append(x)
        g = x
        for n, m in self.up_path._modules.items():
            if int(n) % 2 != 0:
                fine_grained_feature = fine_grained_features.pop()
                crop_size = int((fine_grained_feature.size()[3] - x.size()[3]) / 2)
                concatenate_feature = fine_grained_feature[:, :, crop_size:-crop_size, crop_size:-crop_size]
                if self.is_attention:
                    index_ = int(n) // 2
                    concatenate_feature = self.attention_lst[index_](g, concatenate_feature)
                x = t.cat((concatenate_feature, x), dim=1)
                x = m(x)
                g = x
            else:
                x = m(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.g_conv = nn.Sequential(
            nn.Conv2d(in_channels=F_g, out_channels=F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=F_int)
        )
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_channels=F_l, out_channels=F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=F_int)
        )
        self.last_conv = nn.Conv2d(in_channels=F_int, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        """

        :param g: down path second 3 * 3 convolution result
        :param x: Crop feature
        :return:
        """
        g_conv_result = self.g_conv(g)
        x_conv_result = F.adaptive_avg_pool2d(self.x_conv(x), output_size=g.size()[-2:])
        add_result = g_conv_result + x_conv_result
        relu_result = self.relu(add_result)
        last_conv_result = self.last_conv(relu_result)
        sigmoid_result = self.sigmoid(last_conv_result)
        return x * F.interpolate(sigmoid_result, size=(x.size()[-2:]))


if __name__ == "__main__":
    model = Unet(3, 2, True)
    input_ = t.randn(2, 3, 572, 572)
    print(model(input_).size())
