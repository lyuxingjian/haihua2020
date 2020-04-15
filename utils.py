import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import Optimizer

import warnings
import numpy as np 

from mish_cuda import MishCuda
from efficientnet_pytorch.utils import Swish, MemoryEfficientSwish
from timm.models.layers.activations import Mish, SwishJitAutoFn, MishJitAutoFn
from timm.models.layers.activations import Swish as Swish_timm 

# The single-criterion function fn must accept reduction!
def mix_criterion(preds, labels, fn, reduction='none'):
    labels, labels_shuffled, lam = labels[0], labels[1], labels[2]
    loss = lam * fn(preds, labels, reduction='none') + \
                (1 - lam) * fn(preds, labels_shuffled, reduction='none')
    if reduction == 'none':
        return loss
    else:
        return loss.mean()

### Transform model to mish ###
def to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, (nn.ReLU, Swish, MemoryEfficientSwish, Mish, SwishJitAutoFn, MishJitAutoFn, Swish_timm)):
            setattr(model, child_name, MishCuda())
        else:
            to_mish(child)

# Traceable mish for jit export
class trace_Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x*torch.tanh(F.softplus(x))

def to_tracemish(model):
    for child_name, child in model.named_children():
        if isinstance(child, MishCuda):
            setattr(model, child_name, trace_Mish())
        else:
            to_tracemish(child)

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
        
### My implementation of gridmask
import cv2 
from albumentations import Rotate

# Generate mask for gridmask based on parameters: most important parameter: ratio
# Returns wxh. Samples dimension ratio of mask from (d1, d2)
def generate_mask(w=256, h=256, d1=96/224, d2=1., rotate=45, ratio=.3):
    if ratio == 1.:
        return torch.tensor(np.ones([h, w])*1.)
    
    # h, w = img.shape[0], img.shape[1]
    d1, d2 = int(d1*min(h, w)), int(d2*min(h, w))
    hh, ww = int(1.5*h), int(1.5*w)
    d = np.random.randint(d1, d2)
    l = int(d*ratio+0.5)
    mask = np.ones((hh, ww), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    for i in range(-1, hh//d+1):
        s = d*i + st_h
        t = s+l
        s = max(min(s, hh), 0)
        t = max(min(t, hh), 0)
        mask[s:t,:] *= 0
    for i in range(-1, ww//d+1):
        s = d*i + st_w
        t = s+l
        s = max(min(s, ww), 0)
        t = max(min(t, ww), 0)
        mask[:,s:t] *= 0
    r = np.random.randint(rotate)
    mask = Rotate(limit=rotate, border_mode=cv2.BORDER_CONSTANT,\
                    p=1)(image=mask)['image']
    mask = torch.tensor(mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w])
    mask = 1 - mask
    return mask

### Mixup & Cutmix ###
def mixup(x, label, alpha=.4):
    indices = torch.randperm(x.size(0))
    x_shuffled, label_shuffled = x[indices], label[indices]

    lam = torch.distributions.Beta(alpha, alpha).sample((x.size(0),)).unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
    x = x * lam + x_shuffled * (1 - lam)
    targets = [label, label_shuffled, lam.view(-1)]
    return x, targets

# Assuming pytorch tensor input
def cutmix(x, label, alpha=1.):
    B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
    indices = torch.randperm(B)
    label_shuffled = label[indices]

    Lam = []
    ##### Important! Else cutmixing very messily #####
    x_ = x.clone()
    for i in range(B):
        lam = np.random.beta(alpha, alpha)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        x[i, :, bbx1:bbx2, bby1:bby2] = x_[indices[i], :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        Lam.append(1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    Lam = torch.tensor(Lam).cuda()
    # Means that maximum label is inverted; switch these labels so lam > .5 and label is preserved
    invert_mask = (Lam < .5)
    Lam[invert_mask] = 1 - Lam[invert_mask]
    intermediate = label[invert_mask].clone()
    label[invert_mask] = label_shuffled[invert_mask].clone()
    label_shuffled[invert_mask] = intermediate 

    targets = [label, label_shuffled, Lam]
    return x, targets

import math 
class ArcFaceLoss(nn.Module):
    def __init__(self, s=32, m=.4):
        super().__init__()
        self.s = s 
        self.m = m         
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    # Assume pred has already been multiplied
    def forward(self, pred, label, reduction='mean'):
        cosine = pred / self.s 

        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # Really modified cosine with added margin
        cosine = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = cosine * self.s 
        return torch.nn.functional.cross_entropy(output, label, reduction=reduction)