# Haihua 2020 Waste Sorting Challenge Youth Track 1st Place Solution

Thanks to Haihua for hosting such an [interesting challenge](https://www.biendata.com/competition/haihua_wastesorting_task1)! This is an ideal image classification problem with clean labels, so deep learning methods do well. 

## General
- Resolution: 384x384, plain resize (General_preprocessing.ipynb)
- Optimizer: [Lookahead](https://github.com/alphadl/lookahead.pytorch/blob/master/lookahead.py) + SGD
- OneCycle policy, with 30% warmup and momentum from .8 to .9
- Batch size 64

#### Models
- InceptionV4, efficientnet-b4, and se_resnext50_32x4d from the [Cadene collection](https://github.com/Cadene/pretrained-models.pytorch)
- Mixnet-xl from the [Rwightman collection](https://github.com/rwightman/pytorch-image-models)
- All ReLU, Swish activations converted to [mish](https://github.com/thomasbrandon/mish-cuda)
- Models using pretrained weights from imagenet

#### Augmentations
- Heavy augmentations from [albumentations](https://github.com/albumentations-team/albumentations)
- [Cutmix](https://arxiv.org/abs/1905.04899) on half of the images in each batch
- [Gridmask](https://arxiv.org/abs/2001.04086) on the rest of the images

## Stage 1 Training (.9998)
-Sklearn 5-fold split stratified on labels (64000 train, 16000 validation)

#### [Inference](https://github.com/lyuxingjian/haihua2020/blob/master/Stage1%20Inference.ipynb)
- Dihedral Test time Augmentations (TTA)
- Weights Averaging (WA) using epochs, select based on CV (with TTA)
- Weigh each architecture equally
- [Temperature sharpening](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/107716): .9997 with T=1, .9998 with T=10

#### [Training Schedule](https://github.com/lyuxingjian/haihua2020/blob/master/stage1.sh)
| Model | Fold | LR | Epochs | CV | CV after WA & TTA |
| ------ | :------: | :------: | :------: | :------: | :------: |
| se_resnext50_32x4d | 0 | 0.2 | 20 | .9998 | .9998 |
| se_resnext50_32x4d | 2 | 0.2 | 20 | .9994 | .9996 |
| Efficientnet-b4 | 1 | 0.2 | 30 | .9992 | .9994 |
| Efficientnet-b4 | 3 | 0.2 | 30 | .9993 | .9994 |
| InceptionV4 | 0 | 0.1 | 40 | .9993 | .9998 |
| InceptionV4 | 2 | 0.1 | 40 | .9994 | .9995 |
| InceptionV4 | 3 | 0.1 | 40 | .9996 | .9997 |
| Mixnet-xl | 1 | 0.2 | 30 | .9994 | .9994 |
| Mixnet-xl | 4 | 0.2 | 30 | .9992 | .9996 |

## Stage 2 Training (1.000 with some luck)
- Pseudo-label public test set, and add high-confidence images into train. Confidence threshold: 0.8 (after temperature sharpening); ~9990 qualified images
- Finetune 9 models with pseudo-labeled data added to training set, from weights for stage1 inference
- Ensembling yields 1.000

#### [Training Schedule](https://github.com/lyuxingjian/haihua2020/blob/master/stage2.sh)
| Model | Fold | LR | Epochs |
| ------ | :------: | :------: | :------: |
| se_resnext50_32x4d | 0 | 0.1 | 5 |
| se_resnext50_32x4d | 2 | 0.1 | 5 |
| Efficientnet-b4 | 1 | 0.1 | 8 |
| Efficientnet-b4 | 3 | 0.1 | 8 |
| InceptionV4 | 0 | 0.05 | 10 |
| InceptionV4 | 2 | 0.05 | 10 |
| InceptionV4 | 3 | 0.05 | 10 |
| Mixnet-xl | 1 | 0.1 | 8 |
| Mixnet-xl | 4 | 0.1 | 8 |

#### [Inference](https://github.com/lyuxingjian/haihua2020/blob/master/Stage2_inference.ipynb)
-Plain ensembling, similar to stage 1
