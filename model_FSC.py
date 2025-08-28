from functools import partial

import torch
import torch.nn as nn
from timm.layers import DropPath
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class PermuteToChannelsLast(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

class PermuteToChannelsFirst(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class GCA(nn.Module):
    def __init__(self, dim, expansion_ratio=2.0, kernel_size=3, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 drop_path=0.1):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size,
                              groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.to_channels_last = PermuteToChannelsLast()
        self.to_channels_first = PermuteToChannelsFirst()

    def forward(self, x):
   
        x = self.to_channels_last(x)

        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)

        c = self.to_channels_first(c)
        c = self.conv(c)
        c = self.to_channels_last(c)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))

        out = x + shortcut
        out = self.to_channels_first(out)
        out = self.drop_path(out)

        return out

class FSA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.conv_reduce = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):

        fft = torch.fft.rfft2(x, norm='ortho')
        real, imag = fft.real, fft.imag

        real_avg = self.avg_pool(real)
        real_max = self.max_pool(real)
        real_cat = torch.cat([real_avg, real_max], dim=1)

        imag_avg = self.avg_pool(imag)
        imag_max = self.max_pool(imag)
        imag_cat = torch.cat([imag_avg, imag_max], dim=1)

        real_reduced = self.conv_reduce(real_cat).squeeze(-1).squeeze(-1)
        imag_reduced = self.conv_reduce(imag_cat).squeeze(-1).squeeze(-1)

        real_weights = self.mlp(real_reduced)[:, :, None, None]
        imag_weights = self.mlp(imag_reduced)[:, :, None, None]

        real_attended = real * real_weights
        imag_attended = imag * imag_weights

        fft_attended = torch.complex(real_attended, imag_attended)
        output = torch.fft.irfft2(fft_attended, s=x.shape[-1:], norm='ortho')

        return output

class SPT(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, 5, padding=1, groups=channel)
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, x):
        return x + torch.tanh(self.norm(self.conv(x)))

class DEE_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DEE_module, self).__init__()

        self.FC11 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        self.FC21 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC21.apply(weights_init_kaiming)
        self.FC22 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC22.apply(weights_init_kaiming)
        self.FC23 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC23.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel // 4, channel, kernel_size=1)

        self.FC2.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.01)

        self.gate1 = GCA(dim=channel//reduction, kernel_size=7, conv_ratio=0.5)
        self.gate2 = GCA(dim=channel//reduction, kernel_size=7, conv_ratio=0.5)
        self.spitial1 = SPT(channel=channel//reduction)
        self.spitial2 = SPT(channel=channel//reduction)
        self.freq_attn1 = FSA(channel//reduction)
        self.freq_attn2 = FSA(channel//reduction)

    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x)) / 3
        x1 = self.spitial1(self.gate1(self.freq_attn1(x1)))
        x1 = self.FC1(F.relu(x1))
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x)) / 3
        x2 = self.spitial2(self.gate2(self.freq_attn2(x2)))
        x2 = self.FC2(F.relu(x2))
        out = torch.cat((x, x1, x2), 0)
        out = self.dropout(out)
        return out


class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(high_dim), )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                   nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h

        return z


class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z


class embed_net(nn.Module):
    def __init__(self, class_num, dataset, arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        self.dataset = dataset
        if self.dataset == 'regdb':  # For regdb dataset, we remove the MFA3 block and layer4.
            pool_dim = 1024
            self.DEE = DEE_module(512)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)

        else:
            pool_dim = 2048
            self.DEE = DEE_module(1024)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)
            self.MFA3 = MFA_block(1024, 512, 1)


        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x_ = x
        x = self.base_resnet.base.layer1(x_)
        x_ = self.MFA1(x, x_)
        x = self.base_resnet.base.layer2(x_)
        x_ = self.MFA2(x, x_)
        if self.dataset == 'regdb':  # For regdb dataset, we remove the MFA3 block and layer4.
            x_ = self.DEE(x_)
            x = self.base_resnet.base.layer3(x_)

        else:
            x = self.base_resnet.base.layer3(x_)
            x_ = self.MFA3(x, x_)
            x_ = self.DEE(x_)
            x = self.base_resnet.base.layer4(x_)


        xp = self.avgpool(x)
        x_pool = xp.view(xp.size(0), xp.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            xps = xp.view(xp.size(0), xp.size(1), xp.size(2)).permute(0, 2, 1)
            xp1, xp2, xp3 = torch.chunk(xps, 3, 0)
            xpss = torch.cat((xp2, xp3), 1)
            loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal=1).sum() / (xp.size(0))

            return x_pool, self.classifier(feat), loss_ort
        else:

            return self.l2norm(x_pool), self.l2norm(feat)
