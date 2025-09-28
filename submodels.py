import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class DoubleConv(nn.Module):

    def   __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=kernel//2)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding=kernel//2)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2))
        )

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut
        return x


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# class Evolution_Network(nn.Module):
#     def __init__(self, n_channels, n_classes, base_c=64, bilinear=True):
#         super(Evolution_Network, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         base_c = base_c
#         self.inc = DoubleConv(n_channels, base_c)
#         self.down1 = Down(base_c * 1, base_c * 2)
#         self.down2 = Down(base_c * 2, base_c * 4)
#         self.down3 = Down(base_c * 4, base_c * 8)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(base_c * 8, base_c * 16 // factor)

#         self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
#         self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
#         self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
#         self.up4 = Up(base_c * 2, base_c * 1, bilinear)
#         self.outc = OutConv(base_c * 1, n_classes)
#         self.gamma = nn.Parameter(torch.zeros(1, n_classes, 1, 1), requires_grad=True)

#         self.up1_v = Up(base_c * 16, base_c * 8 // factor, bilinear)
#         self.up2_v = Up(base_c * 8, base_c * 4 // factor, bilinear)
#         self.up3_v = Up(base_c * 4, base_c * 2 // factor, bilinear)
#         self.up4_v = Up(base_c * 2, base_c * 1, bilinear)
#         self.outc_v = OutConv(base_c * 1, n_classes * 2)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x) * self.gamma

#         v = self.up1_v(x5, x4)
#         v = self.up2_v(v, x3)
#         v = self.up3_v(v, x2)
#         v = self.up4_v(v, x1)
#         v = self.outc_v(v)
#         return x, v

class Evolution_Network(nn.Module):
    def __init__(self, n_channels, n_classes, base_c=16, bilinear=True):
        super(Evolution_Network, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c * 1, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        # self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 4, base_c * 8 // factor)

        # self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c * 1, bilinear)
        self.outc = OutConv(base_c * 1, n_classes)
        self.gamma = nn.Parameter(torch.zeros(1, n_classes, 1, 1), requires_grad=True)

        # self.up1_v = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2_v = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3_v = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4_v = Up(base_c * 2, base_c * 1, bilinear)
        self.outc_v = OutConv(base_c * 1, n_classes * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        x5 = self.down4(x3)
        # x = self.up1(x5, x4)
        x = self.up2(x5, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) * self.gamma

        # v = self.up1_v(x5, x4)
        v = self.up2_v(x5, x3)
        v = self.up3_v(v, x2)
        v = self.up4_v(v, x1)
        v = self.outc_v(v)
        return x, v
    
    def load_pretrained_weights(self, pretrained_dict, old_n_classes=4):
        """加载预训练权重，适配n_classes变化"""
        current_dict = self.state_dict()
        
        new_dict = {}
        
        for key, weight in pretrained_dict.items():
            if key in current_dict:
                current_weight = current_dict[key]
                
                # 如果形状匹配，直接使用
                if weight.shape == current_weight.shape:
                    new_dict[key] = weight
                # 处理输出层权重不匹配
                elif 'outc' in key and len(weight.shape) == 4:
                    if 'outc.conv.weight' in key:
                        # 卷积权重 [out_channels, in_channels, H, W]
                        min_classes = min(old_n_classes, self.n_classes)
                        new_weight = current_weight.clone()
                        new_weight[:min_classes] = weight[:min_classes]
                        new_dict[key] = new_weight
                        print(f"调整输出层权重 {key}: {weight.shape} -> {new_weight.shape}")
                    elif 'outc.conv.bias' in key:
                        # 偏置项 [out_channels]
                        min_classes = min(old_n_classes, self.n_classes)
                        new_bias = current_weight.clone()
                        new_bias[:min_classes] = weight[:min_classes]
                        new_dict[key] = new_bias
                        print(f"调整输出层偏置 {key}: {weight.shape} -> {new_bias.shape}")
                # 处理gamma参数（如果有）
                elif 'gamma' in key and len(weight.shape) == 4:
                    min_classes = min(old_n_classes, self.n_classes)
                    new_gamma = current_weight.clone()
                    new_gamma[:, :min_classes] = weight[:, :min_classes]
                    new_dict[key] = new_gamma
                    print(f"调整gamma参数 {key}: {weight.shape} -> {new_gamma.shape}")
                else:
                    # 其他不匹配的情况，跳过并使用随机初始化
                    print(f"跳过不匹配的层 {key}")
            else:
                print(f"跳过不存在的键 {key}")
        
        # 加载权重
        self.load_state_dict(new_dict, strict=False)
        return self