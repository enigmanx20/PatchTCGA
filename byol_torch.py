#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import copy


# In[2]:


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, initializer='original'):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            OrderedDict({
            'ln1': nn.Linear(dim, hidden_dim, bias=True),
            'bn' : nn.BatchNorm1d(hidden_dim),
            'act': nn.ReLU(inplace=True),
            'ln2': nn.Linear(hidden_dim, out_dim, bias=False),       
            })
        )
        if initializer == 'original':
            self._reset_parameters_jax()
    def _reset_parameters_jax(self):
        std = 1. / math.sqrt(self.net.ln1.in_features)
        init._no_grad_trunc_normal_(self.net.ln1.weight, 0.0, std, -1.0, 1.0)
        std = 1. / math.sqrt(self.net.ln2.in_features)
        init._no_grad_trunc_normal_(self.net.ln2.weight, 0.0, std, -1.0, 1.0)
        
        init.zeros_(self.net.ln1.bias)
        
    def forward(self, x):
        return self.net(x)
    
class ResNet50(torchvision.models.ResNet):
    def __init__(self, initializer='original'):
        """
        ResNet50 encoder
        """
        super(ResNet50, self).__init__(block=torchvision.models.resnet.Bottleneck,
                                       layers=[3, 4, 6, 3],
                                       num_classes=1,)
        self.net = nn.Sequential(
                     OrderedDict(
                         {
                             'conv1'  : copy.deepcopy(self.conv1), #self.conv1,
                             'bn1'    : copy.deepcopy(self.bn1), #self.bn1,
                             'act'    : copy.deepcopy(self.relu), #self.relu,
                             'maxpool': copy.deepcopy(self.maxpool), #self.maxpool,
                             'layer1' : copy.deepcopy(self.layer1), #self.layer1,
                             'layer2' : copy.deepcopy(self.layer2), #self.layer2,
                             'layer3' : copy.deepcopy(self.layer3), #self.layer3,
                             'layer4' : copy.deepcopy(self.layer4), #self.layer4,
                         }                                     
                       )
                    )
        del self.conv1
        del self.bn1
        del self.relu
        del self.maxpool
        del self.layer1
        del self.layer2
        del self.layer3
        del self.layer4
        del self.avgpool
        del self.fc
        if initializer == 'original':
            self._reset_parameters_jax()
    def _reset_parameters_jax(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                std = 1. / math.sqrt(fan_in)
                init._no_grad_trunc_normal_(m.weight, 0.0, std, -1.0, 1.0)
                if m.bias:
                    init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        out = self.net(x)
        return out.mean(dim=(2, 3)) # global average pooling

        


# In[ ]:


class BYOL(nn.Module):
    def __init__(self, 
                 encoder_online,
                 encoder_target,
                 projector_hidden_size = 256,
                 predictor_hidden_size = 256,
                 projector_output_size = 256,
                 num_classes=20,
                 initializer='original', **kwargs):
        super(BYOL, self).__init__()
        self.net_online = encoder_online
        self.projector_online = MLP(2048, hidden_dim=projector_hidden_size, out_dim=projector_output_size, initializer=initializer)
        self.predictor_online = MLP(projector_output_size, hidden_dim=predictor_hidden_size, out_dim=projector_output_size, initializer=initializer)
        self.classifier_online = nn.Linear(2048, num_classes)
        
        self.net_target = encoder_target
        self.projector_target = MLP(2048, hidden_dim=projector_hidden_size, out_dim=projector_output_size, initializer=initializer)
        
        if initializer=='original':
            self._reset_parameters_jax()
        
        self._setup_requires_grad()
    
    def _reset_parameters_jax(self):
        std = 1. / math.sqrt(self.classifier_online.in_features)
        init._no_grad_trunc_normal_(self.classifier_online.weight, 0.0, std, -1.0, 1.0)
        
        init.zeros_(self.classifier_online.bias)
        
    def _setup_requires_grad(self):
        for name, param in self.net_online.named_parameters():
            param.requires_grad = True
        for name, param in self.projector_online.named_parameters():
            param.requires_grad = True
        for name, param in self.predictor_online.named_parameters():
            param.requires_grad = True 
        for name, param in self.classifier_online.named_parameters():
            param.requires_grad = True        
           
        for name, param in self.net_target.named_parameters():
            param.requires_grad = False
            param.grad = None
        for name, param in self.projector_target.named_parameters():
            param.requires_grad = False
            param.grad = None
            
    @torch.no_grad()
    def apply_ema(self, tau):
        for param_o, param_t in zip(self.net_online.parameters(), self.net_target.parameters()):
            param_t.data = param_t.data + (1 - tau) * (param_o.data - param_t.data)
        for param_o, param_t in zip(self.projector_online.parameters(), self.projector_target.parameters()):
            param_t.data = param_t.data + (1 - tau) * (param_o.data - param_t.data)
    
    def _online_forward(self, view1, view2):
        embedding_1 = self.net_online(view1)
        proj_out_1 = self.projector_online(embedding_1)
        pred_out_1 = self.predictor_online(proj_out_1)
        logits = self.classifier_online(embedding_1.detach()) # detach classification head from resnet
       
        embedding_2 = self.net_online(view2)
        proj_out_2 = self.projector_online(embedding_2)
        pred_out_2 = self.predictor_online(proj_out_2)
        
        return pred_out_1, pred_out_2, logits
    
    @torch.no_grad()
    def _target_forward(self, view1, view2):
        embedding_1 = self.net_target(view1)
        proj_out_1 = self.projector_target(embedding_1)
       
        embedding_2 = self.net_target(view2)
        proj_out_2 = self.projector_target(embedding_2)
        
        return proj_out_1.detach(), proj_out_2.detach()
        
    def forward(self, view1, view2):
        if not self.training:
            embedding_1 = self.net_online(view1)
            logits = self.classifier_online(embedding_1.detach()) # detach classification out from resnet
            online_out = OrderedDict(
                                  {'logits': logits}
                                )
            return online_out, None
        
        else:
            online_out = OrderedDict(
                                      zip(['pred_out_1', 'pred_out_2', 'logits'], self._online_forward(view1, view2) )
                                    )
            target_out = OrderedDict(
                                      zip(['proj_out_1', 'proj_out_2'], self._target_forward(view1, view2) )
                                    )

            return online_out, target_out





