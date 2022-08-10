
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import BNReLU, CONV, ADAPTIVEAVGPOOL, INTERPOLATE, DROPOUT

class _ASPPModule(nn.Module):
    
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        
        super(_ASPPModule, self).__init__()
        self.conv = CONV(inplanes, planes, kernel_size=kernel_size, stride=1,
                         padding=padding, dilation=dilation, bias=False)
        self.bn = BN(planes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x

class ASPP(nn.Module):
    def __init__(self, inplanes, dilations=[1, 6, 12, 18]):
        super(ASSP, self).__init__()
        
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 1, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 1, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 1, padding=dilations[3], dilation=dilations[3])
        
        self.global_avg_pool = nn.Sequential(ADAPTIVEAVGPOOL(1),
                                             CONV(inplanes, 256, 1, stride=1, bias=False),
                                             BN(256),
                                             nn.ReLU())
        self.conv = CONV(1280, 256, 1, bias=False)
        self.bn = BN(256)
        self.relu = nn.ReLU()
        self.dropout = DROPOUT(0.5)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = INTERPOLATE(self.global_avg_pool(x), size=x.size()[2:], align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        x = self.relu(self.bn(self.conv(x)))
        
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes=256):
        super(Decoder, self).__init__()
        
        self.conv1 = CONV(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BN(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(CONV(256+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BN(256),
                                       nn.ReLU(),
                                       DROPOUT(0.5),
                                       CONV(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BN(256),
                                       nn.ReLU(),
                                       DROPOUT(0.1),
                                       CONV(256, num_classes, kernel_size=1, stride=1))
        
    def forward(self, x, low_level_feat):
        low_level_feat = self.relu(self.bn1(self.conv1(low_level_feat)))
        
        x = INTERPOLATE(x, size=low_level_feat.size()[2:], align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, freeze_bn=False):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = BackboneSelector()()
        num_features = self.backbone.get_num_features()
        
        self.aspp = ASPP(num_features)
        self.decoder = Decoder(num_classes)
        
        self.dsn0 = nn.Sequential(
            CONV(num_features, num_features // 2, kernel_size=3, stride=1, padding=1, bias=False),
            BN(num_features // 2),
            nn.ReLU(),
            DROPOUT(0.1),
            CONV(num_features // 2, num_classes, 1, 1, 0)
        )
        
        self.dsn1 = nn.Sequential(
            CONV(num_features // 2, num_features // 4, kernel_size=3, stride=1, padding=1, bias=False),
            BN(num_features // 4),
            nn.ReLU(),
            DROPOUT(0.1),
            CONV(num_features // 4, num_classes, 1, 1, 0)
        )
        
        if freeze_bn:
            self.freeze_bn()
        
    def forward(self, x_):
        feats = self.backbone(x_)
        aux_x1 = self.dsn1(feats[-2])
        aux_x1 = INTERPOLATE(aux_x1, x_.size[2:], align_corners=True)
        x = self.aspp(feats[-1])
        aux_x0 = self.dsn0(x)
        aux_x0 = INTERPOLATE(aux_x0, x_.size[2:], align_corners=True)
        x = self.decoder(x, feats[0])
        x = INTERPOLATE(x, size=x_.size()[2:], align_corners=True)
        
        return x, aux_x0, aux_x1
        
        
