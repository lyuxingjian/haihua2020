#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

import os
import cv2
import math 
import timm 
import shutil
import argparse 
import numpy as np
import pandas as pd
import pretrainedmodels
import matplotlib.pyplot as plt
import seaborn as sns

from apex import amp
from time import time
from tqdm import tqdm
from warnings import filterwarnings
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
parser.add_argument('--model_name', type=str, default='mixnet_s', help='Model name')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--fold', type=int, default=0, help='Which fold')
parser.add_argument('--total_epochs', type=int, default=16, help='Number of epochs for training')
parser.add_argument('--accum_steps', type=int, default=1, help='Gradient Accumulation?')
parser.add_argument('--pretrained', type=str, default='none', help='path to pretrained backbone, otherwise "none" to imgnet pretrained')
parser.add_argument('--img_root', type=str, default='data/train_384', help='Path to image folder')
parser.add_argument('--pseudo_img_root', type=str, default='data/test_384', help='Path to preprocessed public test images')
parser.add_argument('--mish', type=int, default=1, help='Convert model to mish?')

parser.add_argument('--gpu_no', type=int, default=0, help='GPU index')
parser.add_argument('--lr', type=float, default=5e-2, help='Maximum learning rate')
parser.add_argument('--cutmix_prob', type=float, default=.5, help='Portion of cutmix')
parser.add_argument('--gridmask_ratio', type=float, default=1., help='Gridmask applied ratio')
parser.add_argument('--n_cpus', type=int, default=12, help='Number of workers for dataloading')
parser.add_argument('--val_batch_size', type=int, default=None, help='Batch size')
parser.add_argument('--verbose', type=bool, default=False, help='Be like olaf?')
parser.add_argument('--nowarnings', type=bool, default=True, help='Suppress warnings')

args = parser.parse_args()
args = vars(args)
print(args)


# In[2]:


# args = vars(parser.parse_args())
# args = vars(args)
# print(args)

### Warnings and preliminary settings
os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_no'])
if args['nowarnings']:
    filterwarnings('ignore')
if not os.path.isdir('experiments'):
    os.mkdir('experiments')
logging_folder = os.path.join('experiments', args['experiment_name'])
if os.path.isdir(logging_folder):
    shutil.rmtree(logging_folder)
os.mkdir(logging_folder)
writer = SummaryWriter(logging_folder)
print('For visualization, run:')
print('tensorboard --logdir='+logging_folder)


# In[3]:


'''model portion'''
from mish_cuda import MishCuda
from utils import to_mish, to_tracemish

class add_tail(nn.Module):
    def __init__(self, backbone, num_features):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(num_features, 204)
    
    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)

# Plain model loading
def get_model(args):
    total_names = ['efficientnet-b'+str(i) for i in range(8)] + pretrainedmodels.model_names + timm.list_models()
    if not args['model_name'] in total_names:
        print('Nope! Available models are:', total_names)

    # Load from pretrained first
    if 'efficientnet' in args['model_name']:
        try:
            backbone = EfficientNet.from_pretrained(args['model_name'], 10)
        except:
            print('efficientnet-bx x~[0-7] please')
            raise NotImplementedError
        num_features = backbone._fc.weight.shape[1]
        backbone._fc = nn.Sequential()
    elif args['model_name'] in pretrainedmodels.model_names:
        backbone = pretrainedmodels.__dict__[args['model_name']](pretrained='imagenet')
        num_features = backbone.last_linear.weight.shape[1]
        backbone.last_linear = nn.Sequential()
    else:
        backbone = timm.create_model(args['model_name'], pretrained=True)
        for child_name, child in list(backbone.named_children())[::-1]:
            if isinstance(child, nn.Linear):
                num_features = child.weight.shape[1]
                setattr(backbone, child_name, nn.Sequential())
                break
    model = add_tail(backbone, num_features)
    if args['mish'] == 1:
        to_mish(model)
    return model

model = get_model(args).cuda()
if args['pretrained'] != 'none' and args['pretrained'] != 'None':
    model.load_state_dict(torch.jit.load(args['pretrained'], map_location='cpu').state_dict())


# In[4]:


from PIL import Image
from albumentations import *
from albumentations.pytorch import ToTensorV2
from utils import generate_mask
from functools import partial

class TrashDataset(Dataset):
    def __init__(self, df, root, transforms=ToTensorV2()):
        self.df, self.root = df, root
        self.len = len(df)
        self.transforms = transforms
        self.mask_generator = partial(generate_mask, ratio=args['gridmask_ratio'])
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        label = self.df.label.values[idx]
        name = str(self.df.image_id.values[idx])
        # Reads BGR to RGB hehehe
        img = cv2.imread(os.path.join(self.root, name+'.png'))
        # plt.imshow(img)
        # plt.show()
        img = self.transforms(image=img)['image']
        mask = self.mask_generator(w=img.size(-1), h=img.size(-2)).unsqueeze(0)
        return {'img':img, 'label':label, 'mask':mask}


# In[5]:


# Baseline transforms
train_transform = Compose([
    Flip(p=.75),
    OneOf([
        RandomBrightnessContrast(),
        RandomGamma(),
        HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=50),
    ]),
    OneOf([
        GaussianBlur(),
        Blur(),
        MedianBlur(),
    ]),
    OneOf([
        RandomResizedCrop(384, 384, scale=(.5, 1.), p=.25),
        ShiftScaleRotate(scale_limit=.2, p=.75),
    ]),
    Normalize(),
    ToTensorV2(),
])

val_transform = Compose([
    Normalize(),
    ToTensorV2(),
])

df = pd.read_csv('data/train.csv')
trainset, valset = TrashDataset(df, args['img_root'], train_transform), TrashDataset(df, args['img_root'], val_transform)
indices = np.array(range(len(trainset)))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
for i, (train_indices, val_indices) in enumerate(skf.split(indices, trainset.df.label.values)):
    if i == args['fold']:
        break
trainset, valset = Subset(trainset, train_indices), Subset(valset, val_indices)

# Now add pseudo-labeled dataset to original
pseudo_df = pd.read_csv('checkpoints/stage1_pseudolabels.csv')
pseudo_df['image_id'] = pseudo_df['id']
# Shift by one to start with 0
pseudo_df['label'] -= 1
pseudolabel_trainset = TrashDataset(pseudo_df, args['pseudo_img_root'], train_transform)
trainset = ConcatDataset([trainset, pseudolabel_trainset])
print(len(trainset), 'train', len(valset), 'val')


# In[6]:


'''Data portion'''
# Define trainloader and testloader
def get_loader(args):
    trainloader = DataLoader(trainset, batch_size=args['batch_size'],
                             num_workers=args['n_cpus'], shuffle=True, drop_last=True)
    try:
        val_batch_size = int(args['val_batch_size'])
    except:
        val_batch_size = args['batch_size']
    valloader = DataLoader(valset, shuffle=False, drop_last=False, batch_size=val_batch_size, num_workers=args['n_cpus'])
    return trainloader, valloader

'''load data'''
trainloader, valloader = get_loader(args)

'''Criteria portion'''
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix

# Assumes both pytorch tensors, returns accuracy
def metric(preds, labels):
    preds = preds.argmax(1)
    return (preds == labels).float().mean().item()

def criterion(preds, labels, reduction='mean'):
    return F.cross_entropy(preds, labels, reduction=reduction)

'''Optimizer portion'''
from optim import Ranger, Lookahead, LookAhead_OneCycleLR

op = Lookahead(torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=.9, nesterov=True, weight_decay=5e-4))
total_steps = int(args['total_epochs']*len(trainloader))
scheduler = LookAhead_OneCycleLR(op, max_lr=args['lr'], total_steps=total_steps, base_momentum=.8, max_momentum=.9, pct_start=.3)
model, op = amp.initialize(model, op, opt_level='O2', verbosity=0)


# In[ ]:

'''training loop'''
from utils import cutmix, mix_criterion
i0 = range(args['total_epochs'])
if args['verbose'] == 1:
    i0 = tqdm(i0)

train_loss, train_metric, val_loss, val_metric, lr = [], [], [], [], []
global_steps = 0

for epoch in i0:
    init_time = time()
    
    epoch_train_loss, epoch_train_metric, epoch_val_loss, epoch_val_metric = [], [], [], []
    i1 = trainloader
    if args['verbose'] == 1:
        i1 = tqdm(i1)
    
    # Training 
    model.train()
    for batch in i1:
        x, labels, masks = batch['img'].cuda(), batch['label'].cuda(), batch['mask'].cuda()

        # Cutmix
        pivot = int(x.size(0)*args['cutmix_prob'])
        x[:pivot], targets = cutmix(x[:pivot], labels[:pivot])
        if args['gridmask_ratio'] < 1.:
            x[pivot:] *= masks[pivot:]

        preds = model(x)
        loss_cutmix = mix_criterion(preds[:pivot], targets, criterion, reduction='none')
        loss = criterion(preds[pivot:], labels[pivot:], reduction='none')

        with amp.scale_loss(torch.cat([loss_cutmix, loss]).mean() / args['accum_steps'], op) as scaled_loss:
            scaled_loss.backward()
        if (global_steps + 1) % args['accum_steps'] == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            op.step()
            op.zero_grad()
        scheduler.step()
            
        global_steps += 1
        if global_steps % 10 == 0:
            metric_value = metric(preds, labels)
            epoch_train_metric.append(metric_value)
            lr.append(np.max([group['lr'] for group in op.param_groups]))
            epoch_train_loss.append(loss.mean().item())
            train_loss_value = epoch_train_loss[-1]
            log_dict = {'train':train_loss_value}
            writer.add_scalars('losses', log_dict, global_steps)
            writer.add_scalars('metrics', {'train':metric_value}, global_steps)
            writer.add_scalar('lr', lr[-1], global_steps)
            # writer.add_scalar('scale', model.fc.s.item(), global_steps)
            if args['verbose'] == 1:
                i1.set_postfix({'loss':np.mean(epoch_train_loss), 'metric':np.mean(epoch_train_metric), 'lr': lr[-1]})
    train_loss.extend(epoch_train_loss)
    train_metric.append(np.mean(epoch_train_metric))

    epoch_val_metric = []
    l, p = [], []
    op._backup_and_load_cache()
    model.eval()
    with torch.no_grad():
        for batch in valloader:
            x, labels = batch['img'].cuda(), batch['label'].cuda()
            preds = model(x)
            loss = criterion(preds, labels, reduction='mean')
            epoch_val_metric.append(metric(preds, labels))
            epoch_val_loss.append(loss.item())
            
            l.extend(labels.cpu().numpy())
            p.extend(preds.argmax(1).cpu().numpy())
                
        val_loss.append(np.mean(epoch_val_loss))
        epoch_val_metric = np.mean(epoch_val_metric)
        val_metric.append(epoch_val_metric)
        
        writer.add_scalars('losses', {'val':np.mean(epoch_val_loss)}, global_steps)
        writer.add_scalars('metrics', {'val':epoch_val_metric}, global_steps)
        
        print(f'{epoch+1} loss:{round(np.mean(epoch_train_loss), 4)} metric:{round(np.mean(epoch_train_metric), 4)} '                f'val_loss:{round(val_loss[-1], 4)} val_metric:{round(val_metric[-1], 4)} time:{round(time()-init_time, 4)}')
        init_time = time()

        # Tracing, a little bit more complicated
        model_ = get_model(args).cuda()
        model_.load_state_dict(model.state_dict())
        to_tracemish(model_)
        model_.eval()
        dummy = x.cuda()
        ckpt = torch.jit.trace(model_, dummy)
        torch.jit.save(ckpt, os.path.join(logging_folder, str(epoch)+'.pt'))
        if val_metric[-1] == max(val_metric):
            print('Best total')
            torch.jit.save(ckpt, os.path.join(logging_folder, args['experiment_name']+'.pt'))
    op._clear_and_load_backup()