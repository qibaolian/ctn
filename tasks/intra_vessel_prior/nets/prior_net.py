from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce

from tasks.aneurysm.nets.resunet import norm, CB, CBR, BasicBlock, DownSample, DANetHead, DACBlock, SPPBlock

class DASEResPriorNet18(nn.Module):

    def __init__(self, segClasses=2, k=16, input_channels=1, attention=True, heatmap=False, train_prior=False):

        super(DASEResPriorNet18, self).__init__()
        self.segClasses = segClasses
        self.attention = attention
        self.heatmap = heatmap
        self.train_prior = train_prior

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_channels, k, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock, k, 2, kernel_size=(3, 3, 3), stride=1)
        self.layer2 = self._make_layer(BasicBlock, 2 * k, 2, kernel_size=(3, 3, 3), stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4 * k, 2, kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.layer4 = self._make_layer(BasicBlock, 8 * k, 2, kernel_size=(1, 3, 3), stride=(1, 2, 2))

        if attention:
            self.class4 = DANetHead(8 * k, 8 * k)

        self.class3 = Decoder(8 * k + 4 * k, 4 * k, k)
        self.class2 = Decoder(2 * k + k, 2 * k, k)
        self.class1 = Decoder(k + k, 2 * k, k)

        self.class0 = nn.Sequential(
            nn.Conv3d(4 * k, 2 * k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2 * k, segClasses, kernel_size=1, bias=False)
        )

        self.prior_conv = nn.Sequential(
            BasicBlock(segClasses, 32, stride=2),
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 1, stride=2)
        )
        self.prior_fc1 = nn.Sequential(
            nn.Linear(4096, 1024)
        )
        self.prior_fc2 = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(True)
        )
        self.prior_deconv = nn.Sequential(
            PriorDecoder(1, 64),
            PriorDecoder(64, 32),
            nn.ConvTranspose3d(32, segClasses, kernel_size=4, stride=2, padding=1)
        )

        if self.heatmap:
            self.class_hp = nn.Sequential(
                nn.Conv3d(4 * k, 2 * k, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ELU(inplace=True),
                nn.Conv3d(2 * k, 1, kernel_size=1, bias=False),
                # nn.ReLU(inplace=True),
            )

        # if self.train_prior is False:
        #     p2.weight.requires_grad

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        if self.train_prior:
            seg = self.forward_prior(y, use_onehot=True, only_enc=False)
            return {'y': seg}
        else:
            seg = self.forward_seg(x)
            prob = torch.softmax(seg, dim=1)
            code_seg = self.forward_prior(prob, use_onehot=False, only_enc=True)
            code_y = self.forward_prior(y, use_onehot=True, only_enc=True)
            # return {'y': seg}
            return {'y': seg, 'code_seg': code_seg, 'code_y': code_y}

    def forward_seg(self, x):
        x_size = x.size()
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)

        if self.attention:
            d4 = self.class4(self.layer4(e3))
        else:
            d4 = self.layer4(e3)

        d3 = self.class3(d4, e3)
        d2 = self.class2(d3, e2)
        d1 = self.class1(d2, e1)

        feat = torch.cat([
            e0,
            d1,
            F.interpolate(d2, e0.size()[2:], mode='trilinear', align_corners=True),
            F.interpolate(d3, e0.size()[2:], mode='trilinear', align_corners=True)], 1)

        seg = self.class0(feat)
        seg = F.interpolate(seg, x_size[2:], mode='trilinear', align_corners=True)

        # if self.heatmap:
        #     hp = self.class_hp(feat)
        #     hp = F.interpolate(hp, x_size[2:], mode='trilinear', align_corners=True)
        #     return {'y': seg, 'hp': hp}

        # return {'y': seg}
        return seg

    def forward_prior(self, y, use_onehot=False, only_enc=False):
        if use_onehot:
            batch_size, depth, height, width = y.size()
            y_onehot = torch.zeros(batch_size, self.segClasses, depth, height, width).cuda()
            for batch_index in range(batch_size):
                for class_index in range(0, self.segClasses):
                    y_onehot[batch_index, class_index, y[batch_index] == class_index] = 1
            y = y_onehot
        y = self.prior_conv(y)
        code = y
        # b, c, d, h, w = y.size()
        # y = torch.flatten(y, start_dim=1)
        # code = self.prior_fc1(y)
        # if only_enc:
        #     return code
        # code = self.prior_fc2(code)
        # code = code.view(b, c, d, h, w)
        # from IPython import embed; embed()
        y_rec = self.prior_deconv(code)
        return y_rec

    def _make_layer(self, block, planes, blocks, kernel_size=(3, 3, 3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)


class Decoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = CBR(in_channels, channels)
        self.conv2 = CBR(channels, out_channels)
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        if e is not None:
            x = torch.cat([e,
                           F.interpolate(x, e.size()[2:], mode='trilinear', align_corners=True)], 1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)

        x = self.conv1(x)
        x = self.conv2(x)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)

        x = g1 * x + g2 * x
        return x


class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = CB(out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.sigmoid(x)
        return x


class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = CBR(out_channels, out_channels // 2)
        self.conv2 = CB(out_channels // 2, out_channels)

    def forward(self, x):
        x = nn.AvgPool3d(x.size()[2:])(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x

class PriorDecoder(nn.Module):

    def __init__(self, in_channels, channels, kernel_size=2, stride=2, padding=0):
        super(PriorDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose3d(
            in_channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv2 = CBR(channels, channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x