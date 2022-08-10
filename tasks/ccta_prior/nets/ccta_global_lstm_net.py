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
from tasks.ccta_prior.nets.coord_conv import CoordConv3d


class DASEResLstmNet18(nn.Module):

    def __init__(self, segClasses=2, k=16, attention=True, heatmap=False, use_2d_map=False, use_ct=False):

        super(DASEResLstmNet18, self).__init__()
        self.segClasses = segClasses
        self.attention = attention
        self.heatmap = heatmap
        self.use_ct = use_ct

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(self.segClasses+self.use_ct, k, kernel_size=7, stride=2, padding=3, bias=False)),
            # ('conv1', CoordConv3d(self.segClasses, k, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock, k, 2, kernel_size=(3, 3, 3), stride=1)
        self.layer2 = self._make_layer(BasicBlock, 2 * k, 2, kernel_size=(3, 3, 3), stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4 * k, 2, kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.layer4 = self._make_layer(BasicBlock, 8 * k, 2, kernel_size=(3, 3, 3), stride=(2, 2, 2))

        if attention:
            self.class4 = DANetHead(8 * k, 8 * k)

        self.use_2d_map = use_2d_map
        if self.use_2d_map:
            self.skip1 = SkipLayer(k)
            self.skip2 = SkipLayer(2*k)
            self.skip3 = SkipLayer(4*k)

        self.class3 = Decoder(8 * k + 4 * k, 4 * k, k)
        self.class2 = Decoder(2 * k + k, 2 * k, k)
        self.class1 = Decoder(k + k, 2 * k, k)

        self.class0 = nn.Sequential(
            nn.Conv3d(4 * k, 2 * k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2 * k, segClasses, kernel_size=1, bias=False)
        )

        if self.heatmap:
            self.class_hp = nn.Sequential(
                nn.Conv3d(4 * k, 2 * k, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ELU(inplace=True),
                nn.Conv3d(2 * k, 1, kernel_size=1, bias=False),
                # nn.ReLU(inplace=True),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = input['erase']
        x, gate = self.one_hot(x)
        if self.use_ct:
            image = input['image']
            x = torch.cat((x, image), dim=1)

        seg = self.forward_seg(x)
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
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)

        if self.attention:
            d4 = self.class4(self.layer4(e3))
        else:
            d4 = self.layer4(e3)

        if self.use_2d_map:
            e1 = self.skip1(e1)
            e2 = self.skip2(e2)
            e3 = self.skip3(e3)

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
        y_rec = self.prior_deconv(code)
        return y_rec

    def _make_layer(self, block, planes, blocks, kernel_size=(3, 3, 3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)


class SkipLayer(nn.Module):
    def __init__(self, k, use_concat=True):
        super(SkipLayer, self).__init__()
        self.use_concat = use_concat
        self.rnn = self._mask_rnn_layer(k)
        merge_multiplier = 4 if self.use_concat else 3
        self.merge = nn.Sequential(
            nn.Conv3d(merge_multiplier*k, k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        self.flatten_parameters()
        p0 = self.project0(self.rnn['0'], x)
        p1 = self.project1(self.rnn['1'], x)
        p2 = self.project2(self.rnn['2'], x)
        if self.use_concat:
            p = torch.cat([x, p0, p1, p2], 1)
        else:
            p = torch.cat([p0, p1, p2], 1)
        x = self.merge(p)
        return x

    def _mask_rnn_layer(self, k):
        assert k%2 == 0
        rnn = nn.ModuleDict()
        for key in ['0', '1', '2']:
            rnn_k = nn.ModuleList([
                nn.LSTM(k, k // 2, bidirectional=True),
                nn.LSTM(k, k // 2, bidirectional=True)
            ])
            rnn[key] = rnn_k
        return rnn

    def flatten_parameters(self):
        for k, v in self.rnn.items():
            for r in v:
                r.flatten_parameters()

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

class Decoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = CBR(in_channels, channels)
        self.conv2 = CBR(channels, out_channels)
        self.spatial_gate = sSE(out_channels)
        # self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        if e is not None:
            x = torch.cat(
                [e, F.interpolate(x, e.size()[2:], mode='trilinear', align_corners=True)], 1
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)

        x = self.conv1(x)
        x = self.conv2(x)

        g1 = self.spatial_gate(x)
        # g2 = self.channel_gate(x)

        # x = g1 * x + g2 * x
        x = g1 * x
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