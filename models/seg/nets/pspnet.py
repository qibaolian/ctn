import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import BNReLU, CONV, ADAPTIVEAVGPOOL, INTERPOLATE, DROPOUT

class CBRBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=1, dilation=1):
        super(CBRBlock, self).__init__()
        self.conv = CONV(in_channels=inplanes,out_channels=outplanes,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation = dilation, bias=False)
        self.bn_relu = BNReLU(outplanes)
    
    def forward(self, x):
        x = self.bn_relu(self.conv(x))
        return x

#PSP decoder Part
#pyramaid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    def __init__(self, fc_dim=4096):
        super(PPMBilinearDeepsup, self).__init__()
        pool_scales = (1, 2, 3, 6)
        self.ppm = []
        
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                ADAPTIVEAVGPOOL(scale),
                CONV(fc_dim, 512, kernel_size=1, bias=False),
                BNReLU(512)
            ))
        
        self.ppm = nn.ModuleList(self.ppm)
    
    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pool_scale in self.ppm:
            ppm_out.append(INTERPOLATE(pool_scale(x), input_size[2:], 
                                       align_corners=True))
        
        ppm_out = torch.cat(ppm_out, 1)
        return ppm_out
    
        
class PSPNet(nn.Sequential):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        self.backbone = BackboneSelector()()
        num_features = self.backbone.get_num_features()
        
        self.dsn = nn.Sequential(
            CBRBlock(num_features // 2, num_features // 4, 3, 1),
            DROPOUT(0.1),
            CONV(num_features // 4, num_classes, 1, 1, 0)
        )
        
        self.ppm = PPMBilinearDeepsup(fc_dim=num_features)
        
        self.cls = nn.Sequential(
            CONV(num_features + 4 * 512, 512, kernel_size=3, padding=1, bias=False),
            BNReLU(512),
            DROPOUT(0.1),
            CONV(512, num_classes, kernel_size=1)
        )
        
    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.ppm(x[-1])
        x = self.cls(x)
        aux_x = INTERPOLATE(aux_x, x_.size[2:], align_corners=True)
        x = INTERPOLATE(x, x_size[2:], align_corners = True)
        return x, aux_x

            
        