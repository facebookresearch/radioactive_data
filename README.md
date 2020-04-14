# Radioactive data

This is the open source implementation of the paper "Radioactive data: tracing through training".
If you use this code, please cite the paper

```
@article{sablayrolles2020radioactive,
  title={Radioactive data: tracing through training},
  author={Sablayrolles, Alexandre and Douze, Matthijs and Schmid, Cordelia and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:2002.00937},
  year={2020}
}
```

## Install

The install only requires Numpy and Pytorch 1.0
```python
conda install numpy
# See http://pytorch.org for details
conda install pytorch -c pytorch
```

## Creating radioactive data

First, specify a marking model
```python
import torch
from torchvision import models

resnet18 = models.resnet18(pretrained=True)
torch.save({
    "model": resnet18.state_dict(),
    "params": {
      "architecture": "resnet18",
    }
  }, "pretrained_resnet18.pth")

```

Then sample random (normalized) directions as carriers:
```python
import torch

n_classes, dim = 1000, 512
carriers = torch.randn(n_classes, dim)
carriers /= torch.norm(carriers, dim=1, keepdim=True)
torch.save(carriers, "carriers.pth")
```

```
python make_data_radioactive.py \
--carrier_id 0 \
--carrier_path carriers.pth \
--data_augmentation random \
--epochs 90 \
--img_paths img1.jpeg,img2.jpeg \
--lambda_ft_l2 0.01 \
--lambda_l2_img 0.0005 \
--marking_network pretrained_resnet18.pth \
--optimizer sgd,lr=1.0
```

## Training a model

To train a model, we need to create a file that lists all the images that have been replaced:

```python
import torch

torch.save({
  'type': 'per_sample',
  'content': {
    988: 'img1_radio.npy',
  }
}, "radioactive_data.pth")
```

We can then launch training:
```
python train-classif.py \
--architecture resnet18 \
--dataset imagenet \
--epochs 90 \
--train_path radioactive_data.pth \
--train_transform random
```

## Detecting if a model is radioactive

```
python detect_radioactivity.py \
--carrier_path carriers.pth \
--marking_network pretrained_resnet18.pth \
--tested_network /checkpoint/asablayrolles/radioactive_data/train_imagenet_whole_percentage_bs/_train_path=10_train_transform=random/checkpoint-0.pth
```

## License

This repository is licensed under the CC BY-NC 4.0.
