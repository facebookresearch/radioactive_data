# Radioactive data

This is the open source implementation of the paper ["Radioactive data: tracing through training"](https://arxiv.org/abs/2002.00937).

Radioactive data can be used to detect whether a particular image dataset has been used to train a model. 
It makes imperceptible changes to this dataset such that any model trained on it will bear an identifiable mark. The mark is robust to strong variations such as different network architectures or optimization methods. Given a trained model, our technique detects the use of radioactive data and provides a level of confidence (p-value). 

## Install

The install only requires Numpy and Pytorch >= 1.0
```python
conda install numpy
# See http://pytorch.org for details
conda install pytorch -c pytorch
```

## Creating radioactive data

Marking the data is very easy.
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

The `make_data_radioactive.py` script does the actual marking.
For example, to mark images `img1.jpeg` and `img2.jpeg` with the carrier #0 (which should be the same as the class id), do:
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
--dump-path /path/to/images \
--optimizer sgd,lr=1.0
```

This takes about 1 minute and the output images are stored in /path/to/images.


## Training a model

The training is not controlled by the adversary. 
However, to simulate it, we perform a standard training as follows.

To train a model, we need to create a file that lists all the images that have been replaced with marked versions:

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

So now we use the carriers and the trained network to detect the radioactive marks.

```
python detect_radioactivity.py \
--carrier_path carriers.pth \
--marking_network pretrained_resnet18.pth \
--tested_network checkpoint-0.pth
```

On the output, you should obtain a line with "log10(p)=...", which gives the (log of the) p-value of radioactivity detection. 

## Citation

If you use this code, please cite the paper

```
@article{sablayrolles2020radioactive,
  title={Radioactive data: tracing through training},
  author={Sablayrolles, Alexandre and Douze, Matthijs and Schmid, Cordelia and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:2002.00937},
  year={2020}
}
```


## License

This repository is licensed under the CC BY-NC 4.0.
