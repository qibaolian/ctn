
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import CONV, ADAPTIVEAVGPOOL, INTERPOLATE, DROPOUT, F_AVG_POOL
from models.seg.nets.unet import BasicBlock, _TransitionUp

class CBR(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.add_module('conv', CONV(in_channels, out_channels, 
                                     kernel_size=3, stride=stride, padding=1, bias=False))
        self.add_module('bn', BN(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        
class CR(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super(CR, self).__init__()
        self.add_module('conv', CONV(in_channels, out_channels, 
                                     kernel_size=3, stride=stride, padding=1, bias=False))
        self.add_module('relu', nn.ReLU(inplace=True))
        
class BackboneUNet(nn.Module):
    
    def __init__(self, num_classes, deep_supervison=True):
        super(BackboneUNet, self).__init__()
        self.backbone = BackboneSelector()()
        
        self.layer4_1x1 = CR(self.backbone.channels[4], 512, 1, 1, 0)
        self.layer3_1x1 = CR(self.backbone.channels[3], 256, 1, 1, 0)
        self.layer2_1x1 = CR(self.backbone.channels[2], 128, 1, 1, 0)
        self.layer1_1x1 = CR(self.backbone.channels[1], 64, 1, 1, 0)
        self.layer0_1x1 = CR(self.backbone.channels[0], 64, 1, 1, 0)
        
        self.up4 = _TransitionUp(512, 512, 0, 'conv')
        self.up3 = _TransitionUp(256, 256, 0, 'conv')
        self.up2 = _TransitionUp(256, 256, 0, 'conv')
        self.up1 = _TransitionUp(128, 128, 0, 'conv')
        
        self.decoder3 = BasicBlock(256+512, 256)
        self.decoder2 = BasicBlock(128+256, 256)
        self.decoder1 = BasicBlock(64+256, 128)
        self.decoder0 = BasicBlock(64+128, 64)
        
        if self.backbone.first_stride:
            self.feat_orign = BasicBlock(self.backbone.input_channel, 64)
            self.up_orign = _TransitionUp(64, 64, 0, 'conv')
            self.decoder_orign = BasicBlock(64+64, 64)
        
        self.final_conv = CONV(64, num_classes, 1, 1, 0, bias=False)
        
        self.deep_supervison = deep_supervison
        if deep_supervison:
            num_features = self.backbone.get_num_features() 
            self.cls_fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        
        if self.backbone.first_stride:
            feat_orign = self.feat_orign(x)
        
        feats = self.backbone(x)
        
        if self.deep_supervison:
            x0 = F_AVG_POOL(feats[4], feats[4].size()[2:])
            x0 = x0.view(x0.size(0), -1)
            x0 = self.cls_fc(x0)
            
        x = self.up4(self.layer4_1x1(feats[4]), self.layer3_1x1(feats[3]))
        x = self.decoder3(x)
        
        x = self.up3(x, self.layer2_1x1(feats[2]))
        x = self.decoder2(x)
        
        x = self.up2(x, self.layer1_1x1(feats[1]))
        x = self.decoder1(x)
        
        x = self.up1(x, self.layer0_1x1(feats[0]))
        x = self.decoder0(x)
        
        if self.backbone.first_stride:
            x = self.up_orign(x, feat_orign)
            x = decoder_orign(x)
        
        x = self.final_conv(x)
        
        return x, x0 if self.deep_supervison else x
                     
                                     
        