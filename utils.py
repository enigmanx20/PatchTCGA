import os
import random
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

def sample_CV(dataset, val_split=0.02, test_split=0.01, n_trial=3, patches_per_slide=500):
    import pickle
    kth_trial = {}
    slides_each_organ    = OrderedDict()
        
    for organ_id, imgs in dataset.data_in_each_label.items():
        num_imgs = len( dataset.data_in_each_label[organ_id].keys() )
        num_slides = num_imgs//patches_per_slide
        #print(organ_id, num_imgs, num_slides)
        slides_each_organ[organ_id] = list(range(num_slides))
        random.shuffle(slides_each_organ[organ_id])
    for n in range(n_trial):
        train_idx = []
        train_imgs = []
        val_idx   = []
        val_imgs = []
        test_idx  = []
        test_imgs = []
        for organ_id, slides in slides_each_organ.items():
            num_slides = len( slides )
            num_val = math.ceil(num_slides * val_split)
            num_test = math.ceil(num_slides * test_split)
            val_slides = slides[(num_val+num_test)*n:(num_val+num_test)*n + num_val]
            test_slides = slides[(num_val+num_test)*n + num_val : (num_val+num_test)*n + num_val + num_test]
            counter = -1
            for img, idx in dataset.data_in_each_label[organ_id].items() :
                if idx%500 == 0:
                    counter += 1
                if counter in val_slides:
                    val_idx.append( idx )
                    val_imgs.append(os.path.basename( img ))
                elif counter in test_slides:
                    test_idx.append( idx )
                    test_imgs.append(os.path.basename( img ))
                else:
                    train_idx.append( idx ) 
                    train_imgs.append(os.path.basename( img ))
        kth_trial[str(n)] = {'train': {'idx': train_idx, 'imgs': train_imgs},
                             'val'  : {'idx':   val_idx, 'imgs':   val_imgs},
                             'test' : {'idx':  test_idx, 'imgs':  test_imgs}}
    with open('./{}fold_dict_idx_filenames.pickle'.format(n_trial), 'wb') as f:
        pickle.dump(kth_trial, f)
    return kth_trial

def print_num_parameters(model):
    num_params = 0
    for n, p in model.named_parameters():
        num_params += torch.prod(torch.Tensor(tuple(p.data.shape))).sum().item()
    print('# of parameters: ', num_params//2**20, 'M')

from torchvision.models._utils import IntermediateLayerGetter
import torchvision.models.segmentation.fcn as fcn

def FCN_CNN_resnet(
    backbone,
    num_classes: int,
) -> fcn.FCN:
    return_layers = {"layer4": "out"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = fcn.FCNHead(2048, num_classes)
    aux_classifier = None
    return fcn.FCN(backbone, classifier, aux_classifier)

def FCN_CNN_incv3(
    backbone,
    num_classes: int,
) -> fcn.FCN:
    def encode(x):
        x = backbone.Conv2d_1a_3x3(x)
        x = backbone.Conv2d_2a_3x3(x)
        x = backbone.Conv2d_2b_3x3(x)
        x = backbone.maxpool1(x)
        x = backbone.Conv2d_3b_1x1(x)
        x = backbone.Conv2d_4a_3x3(x)
        x = backbone.maxpool2(x)
        x = backbone.Mixed_5b(x)
        x = backbone.Mixed_5c(x)
        x = backbone.Mixed_5d(x)
        x = backbone.Mixed_6a(x)
        x = backbone.Mixed_6b(x)
        x = backbone.Mixed_6c(x)
        x = backbone.Mixed_6d(x)
        x = backbone.Mixed_6e(x)
        x = backbone.Mixed_7a(x)
        x = backbone.Mixed_7b(x)
        x = backbone.Mixed_7c(x)
        return {'out': x}
        
    backbone.forward = encode
    
    classifier = fcn.FCNHead(2048, num_classes)
    aux_classifier = None
    return fcn.FCN(backbone, classifier, aux_classifier)

def FCN_CNN_eff(
    backbone,
    num_classes: int,
) -> fcn.FCN:
    backbone.classifier[1] = nn.Identity()
    backbone.forward = lambda x: {'out': backbone.forward(x)}
    classifier = fcn.FCNHead(1536, num_classes)
    aux_classifier = None
    return fcn.FCN(backbone, classifier, aux_classifier)

def FCN_CNN_vit(
    backbone,
    num_classes: int,
    n=0,
) -> fcn.FCN:
    backbone.forward = lambda x: {
                                 'out': backbone.get_intermediate_layers(x, n=n, reshape=True, norm=True),
                                 }

    classifier = fcn.FCNHead(backbone.embed_dim, num_classes)
    aux_classifier = None
    return fcn.FCN(backbone, classifier, aux_classifier)


#from Big-GAN
# Utility file to seed rngs
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

# from mocov2
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

import torch
import torch.optim as optim
import warnings
import math
SAVE_STATE_WARNING = "Please also save or load the state of the optimizer when saving or loading the scheduler."

class GradualWarmupWithCosineLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, cold_step, peak_step, max_step, initial_lr=1e-6, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.cold_step = cold_step
        self.peak_step = peak_step
        self.max_step = max_step
        self.initial_lr = initial_lr
        super(GradualWarmupWithCosineLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        warnings.warn(SAVE_STATE_WARNING, UserWarning)
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer')}

        return state_dict


    def load_state_dict(self, state_dict):
        warnings.warn(SAVE_STATE_WARNING, UserWarning)
        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        if self.last_epoch == 0:
            return [self.initial_lr for base_lr in self.base_lrs]
        if self._step_count < self.cold_step:
            return [self.initial_lr for base_lr in self.base_lrs]
        if self._step_count - self.peak_step < 0:
            return [self.initial_lr + (self._step_count-self.cold_step)/ float(self.peak_step-self.cold_step) * (base_lr - self.initial_lr) 
                    for base_lr in self.base_lrs]
        else:
            return [self.initial_lr + (base_lr - self.initial_lr) * (1 + math.cos(math.pi * self._step_count/self.max_step)) / 2
                for base_lr in self.base_lrs]

# from simCLR repo
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class LARS(object):
    """
    Slight modification of LARC optimizer from https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py
    Matches one from SimCLR implementation https://github.com/google-research/simclr/blob/master/lars_optimizer.py
    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the adaptive lr. See https://arxiv.org/abs/1708.03888
    """

    def __init__(self,
                 optimizer,
                 trust_coefficient=0.001,
                 ):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if weight_decay != 0:
                        p.grad.data += weight_decay * p.data

                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)
                    adaptive_lr = 1.

                    if param_norm != 0 and grad_norm != 0 and group['layer_adaptation']:
                        adaptive_lr = self.trust_coefficient * param_norm / grad_norm

                    p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]


