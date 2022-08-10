##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Modified from: https://github.com/AlexHex7/Non-local_pytorch
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import os
import sys
import pdb
from torch import nn
from torch.nn import functional as F
import functools

torch_ver = torch.__version__[:3]   


class _PyramidSelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_PyramidSelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            #InPlaceABNSync(self.key_channels)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv3d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv3d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)


    def forward(self, x):
        batch_size, c, h, w, d = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)

        local_x = []
        local_y = []
        local_z = []
        step_h, step_w, step_d = h//self.scale, w//self.scale, d//self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                for k in range(0, self.scale):
                    start_x, start_y, start_z = i*step_h, j*step_w, k*step_d
                    end_x, end_y, end_z = min(start_x+step_h, h), min(start_y+step_w, w), min(start_z+step_d, d)
                    if i == (self.scale-1):
                        end_x = h
                    if j == (self.scale-1):
                        end_y = w
                    if k == (self.scale-1):
                        end_z = d
                    local_x += [start_x, end_x]
                    local_y += [start_y, end_y]
                    local_z += [start_z, end_z]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        local_list = []
        local_block_cnt = 2*self.scale*self.scale*self.scale
        for i in range(0, local_block_cnt, 2):
            value_local = value[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1], local_z[i]:local_z[i+1]]
            query_local = query[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1], local_z[i]:local_z[i+1]]
            key_local = key[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1], local_z[i]:local_z[i+1]]

            h_local, w_local, d_local = value_local.size(2), value_local.size(3), value_local.size(4)
            value_local = value_local.contiguous().view(batch_size, self.value_channels, -1)
            value_local = value_local.permute(0, 2, 1)

            query_local = query_local.contiguous().view(batch_size, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.key_channels, -1)

            sim_map = torch.matmul(query_local, key_local)
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.matmul(sim_map, value_local)
            context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size, self.value_channels, h_local, w_local, d_local)
            local_list.append(context_local)

        context_list = []
        for i in range(0, self.scale):           
            row_tmp = []
            for j in range(0, self.scale):
                v_tmp = []
                for k in range(0, self.scale):
                    v_tmp.append(local_list[k+(j+i*self.scale)*self.scale])
                row_tmp.append(torch.cat(v_tmp, 4))
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = self.W(context)

        return context


class PyramidSelfAttentionBlock3D(_PyramidSelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(PyramidSelfAttentionBlock3D, self).__init__(in_channels,
                                                            key_channels,
                                                            value_channels,
                                                            out_channels,
                                                            scale)



class Pyramid_OC_Module(nn.Module):
    """
    Output the combination of the context features and the original features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """
    def __init__(self, in_channels, out_channels, dropout, sizes=([1])):
        super(Pyramid_OC_Module, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, in_channels // 2, in_channels, size) for size in sizes])        
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv3d(2*in_channels*self.group, out_channels, kernel_size=1, padding=0),
            #InPlaceABNSync(out_channels),
            nn.Dropout3d(dropout)
            )
        self.up_dr = nn.Sequential(
            nn.Conv3d(in_channels, in_channels*self.group, kernel_size=1, padding=0),
            #InPlaceABNSync(in_channels*self.group)
            )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return PyramidSelfAttentionBlock3D(in_channels,    
                                            key_channels,
                                            value_channels,
                                            output_channels, 
                                            size)
        
    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = [self.up_dr(feats)]
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn_dropout(torch.cat(context, 1))
        return output