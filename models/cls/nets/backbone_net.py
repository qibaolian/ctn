
import torch
import torch.nn as nn

from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import F_AVG_POOL

class BackBoneNet(nn.Module):
    
    def __init__(self, num_classes):
        super(BackBoneNet, self).__init__()
        
        self.backbone = BackboneSelector()()
        num_features = self.backbone.get_num_features()
        
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)[-1]
        
        x = F_AVG_POOL(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet(BackBoneNet):
    
    def __init__(self, num_classes):
        
        #assert 'resnet' in
        super(ResNet, self).__init__(num_classes)

class DenseNet(BackBoneNet):
    
    def __init__(self, num_classes):
        
        #assert 'densenet' in
        super(DenseNet, self).__init__(num_classes)
        
    