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


class DASERes64SmallNet18(nn.Module):

    def __init__(self, segClasses=2, k=16):

        super(DASERes64SmallNet18, self).__init__()
        self.segClasses = segClasses

        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(self.segClasses, k, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(k, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', norm(k)),
            ('relu2', nn.ReLU(inplace=True))
        ]))
        self.inplanes = k
        self.layer2 = self._make_layer(BasicBlock, k, 6, kernel_size=(3, 3, 3), stride=2)

        self.class2 = Decoder(k, k, k)
        self.class1 = Decoder(k, k, segClasses)

        # self.class0 = nn.Sequential(
        #     nn.Conv3d(k, k, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ELU(inplace=True),
        #     nn.Conv3d(k, segClasses, kernel_size=1, bias=False)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x, gate = self.one_hot(x)
        seg = self.forward_seg(x)
        # seg = (1-gate)*x + gate*seg
        seg = x + seg
        return {'y': seg}

    def one_hot(self, x):
        batch_size, c, depth, height, width = x.size()
        xx = torch.zeros(batch_size, self.segClasses, depth, height, width).cuda()
        gate = torch.zeros(batch_size, self.segClasses, depth, height, width).cuda()
        for batch_index in range(batch_size):
            for class_index in range(0, self.segClasses):
                xx[batch_index, class_index, x[batch_index, 0] == class_index] = 1
                gate[batch_index, class_index] = x[batch_index, 0] == 0
        return xx, gate

    def forward_seg(self, x):
        x_size = x.size()
        # e0 = self.layer0(x)
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        # from IPython import embed; embed()
        d2 = self.class2(e2)
        seg = self.class1(d2)

        # seg = self.class0(d1)
        # seg = F.interpolate(seg, x_size[2:], mode='trilinear', align_corners=True)

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
        y_rec = self.prior_deconv(code)
        return y_rec

    def project(self, rnn, merge, x):
        p0 = self.project0(rnn[0], x)
        p1 = self.project1(rnn[1], x)
        p2 = self.project2(rnn[2], x)
        if self.use_concat:
            p = torch.cat([x, p0, p1, p2], 1)
        else:
            p = torch.cat([p0, p1, p2], 1)
        p = merge(p)
        return p

    def project0(self, rnn, x):
        b, c, d, h, w = x.size()
        x = torch.max(x, 2, keepdim=False)[0]  # BxCxHxW
        # from left to right
        x = x.permute(3, 0, 2, 1)  # WxBxHxC
        x = x.reshape(w, b * h, c)
        x = rnn[0](x)[0]
        x = x.reshape(w, b, h, c)  # WxBxHxC
        x = x.permute(2, 1, 0, 3)  # HxBxWxC
        x = x.reshape(h, b * w, c)
        x = rnn[1](x)[0]
        x = x.reshape(h, b, w, c)  # HxBxWxC
        x = x.permute(1, 3, 0, 2)  # BxCxHxW
        x = x.reshape(b, c, 1, h, w)  # BxCxDxHxW
        x = x.expand(-1, -1, d, -1, -1)
        return x

    def project1(self, rnn, x):
        b, c, d, h, w = x.size()
        x = torch.max(x, 3, keepdim=False)[0]  # BxCxDxW
        # from left to right
        x = x.permute(3, 0, 2, 1)  # WxBxDxC
        x = x.reshape(w, b * d, c)
        x = rnn[0](x)[0]
        x = x.reshape(w, b, d, c)  # WxBxDxC
        x = x.permute(2, 1, 0, 3)  # DxBxWxC
        x = x.reshape(d, b * w, c)
        x = rnn[1](x)[0]
        x = x.reshape(d, b, w, c)  # DxBxWxC
        x = x.permute(1, 3, 0, 2)  # BxCxDxW
        x = x.reshape(b, c, d, 1, w)  # BxCxDxHxW
        x = x.expand(-1, -1, -1, h, -1)
        return x

    def project2(self, rnn, x):
        b, c, d, h, w = x.size()
        x = torch.max(x, 4, keepdim=False)[0]  # BxCxDxH
        # from left to right
        x = x.permute(3, 0, 2, 1)  # HxBxDxC
        x = x.reshape(h, b * d, c)
        x = rnn[0](x)[0]
        x = x.reshape(h, b, d, c)  # WxBxDxC
        x = x.permute(2, 1, 0, 3)  # DxBxHxC
        x = x.reshape(d, b * h, c)
        x = rnn[1](x)[0]
        x = x.reshape(d, b, h, c)  # DxBxHxC
        x = x.permute(1, 3, 0, 2)  # BxCxDxH
        x = x.reshape(b, c, d, h, 1)  # BxCxDxHxW
        x = x.expand(-1, -1, -1, -1, w)
        return x

    def _make_layer(self, block, planes, blocks, kernel_size=(3, 3, 3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _mask_rnn_layer(self, k):
        assert k%2 == 0
        rnn0 = nn.ModuleList([
            nn.LSTM(k, k // 2, bidirectional=True),
            nn.LSTM(k, k // 2, bidirectional=True)
        ])
        rnn1 = nn.ModuleList([
            nn.LSTM(k, k // 2, bidirectional=True),
            nn.LSTM(k, k // 2, bidirectional=True)
        ])
        rnn2 = nn.ModuleList([
            nn.LSTM(k, k // 2, bidirectional=True),
            nn.LSTM(k, k // 2, bidirectional=True)
        ])
        rnn = nn.ModuleList([rnn0, rnn1, rnn2])
        return rnn


class Decoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = CBR(in_channels, channels)
        self.conv2 = CBR(channels, out_channels)
        # self.spatial_gate = sSE(out_channels)
        # self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        if e is not None:
            x = torch.cat([e,
                           F.interpolate(x, e.size()[2:], mode='trilinear', align_corners=True)], 1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)

        x = self.conv1(x)
        x = self.conv2(x)

        # g1 = self.spatial_gate(x)
        # g2 = self.channel_gate(x)

        # x = g1 * x + g2 * x
        # x = g1 * x
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