# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
from os.path import basename, join

from src.data_augmentations import RandomResizedCropFlip, CenterCrop
from src.dataset import getImagenetTransform, NORMALIZE_IMAGENET
from src.datasets.folder import default_loader
from src.model import build_model
from src.utils import initialize_exp, bool_flag, get_optimizer, repeat_to

image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1)
image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1)

def numpyTranspose(x):
    return np.transpose(x.numpy(), (1, 2, 0))


def numpyPixel(x):
    pixel_image = torch.clamp(255 * ((x * image_std) + image_mean), 0, 255)
    return np.transpose(pixel_image.numpy().astype(np.uint8), (1, 2, 0))


def roundPixel(x):
    x_pixel = 255 * ((x * image_std) + image_mean)
    y = torch.round(x_pixel).clamp(0, 255)
    y = ((y / 255.0) - image_mean) / image_std
    return y


def project_linf(x, y, radius):
    delta = x - y
    delta = 255 * (delta * image_std)
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / image_std
    return y + delta


def psnr(delta):
    return 20 * np.log10(255) - 10 * np.log10(np.mean(delta**2))


def get_parser():
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument("--dump_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="bypass")
    parser.add_argument("--exp_id", type=str, default="")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--data_augmentation", type=str, default="random", choices=["center", "random"])

    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lambda_ft_l2", type=float, default=0.5)
    parser.add_argument("--lambda_l2_img", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1-0.01-0.001,momentum=0.9,weight_decay=0.0001")
    parser.add_argument("--carrier_path", type=str, default="", help="Direction in which to move features")
    parser.add_argument("--carrier_id", type=int, default=0, help="Id of direction in direction array")

    parser.add_argument("--angle", type=float, default=None, help="Angle (if cone)")
    parser.add_argument("--half_cone", type=bool_flag, default=True)

    parser.add_argument("--img_list", type=str, default=None,
                        help="File that contains list of all images")
    parser.add_argument("--img_paths", type=str, default='',
                        help="Path to image to which apply adversarial pattern")

    parser.add_argument("--marking_network", type=str, required=True)
    # parser.add_argument("--image_sizes")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug from a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    return parser


def main(params):
    logger = initialize_exp(params)

    if params.img_list is None:
        params.img_paths = [s.strip() for s in params.img_paths.split(",")]
    else:
        assert ":" in params.img_paths
        chunks = params.img_paths.split(":")
        assert len(chunks) == 2
        n_start, n_end = int(chunks[0]), int(chunks[1])

        img_list = torch.load(params.img_list)
        params.img_paths = [img_list[i] for i in range(n_start, n_end)]
    print("Image paths", params.img_paths)

    # Build model / cuda
    ckpt = torch.load(params.marking_network)
    params.num_classes = ckpt["params"]["num_classes"]
    params.architecture = ckpt['params']['architecture']
    print("Building %s model ..." % params.architecture)
    model = build_model(params)
    model.cuda()
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt['model'].items()}, strict=False)
    model = model.eval()
    model.fc = nn.Sequential()

    loader = default_loader
    transform = getImagenetTransform("none", img_size=params.img_size, crop_size=params.crop_size)
    img_orig = [transform(loader(p)).unsqueeze(0) for p in params.img_paths]

    # Loading carriers
    direction = torch.load(params.carrier_path).cuda()
    assert direction.dim() == 2
    direction = direction[params.carrier_id:params.carrier_id + 1]

    rho = -1
    if params.angle is not None:
        rho = 1 + np.tan(params.angle)**2

    img = [x.clone() for x in img_orig]

    # Load differentiable data augmentations
    center_da = CenterCrop(params.img_size, params.crop_size)
    random_da = RandomResizedCropFlip(params.crop_size)
    if params.data_augmentation == "center":
        data_augmentation = center_da
    elif params.data_augmentation == "random":
        data_augmentation = random_da

    for i in range(len(img)):
        img[i].requires_grad = True

    optimizer, schedule = get_optimizer(img, params.optimizer)
    if schedule is not None:
        schedule = repeat_to(schedule, params.epochs)

    img_center = torch.cat([center_da(x, 0).cuda(non_blocking=True) for x in img_orig], dim=0)
    # ft_orig = model(center_da(img_orig, 0).cuda(non_blocking=True)).detach()
    ft_orig = model(img_center).detach()

    if params.angle is not None:
        ft_orig = torch.load("/checkpoint/asablayrolles/radioactive_data/imagenet_ckpt_2/features/valid_resnet18_center.pth").cuda()

    for iteration in range(params.epochs):
        if schedule is not None:
            lr = schedule[iteration]
            logger.info("New learning rate for %f" % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Differentially augment images
        batch = []
        for x in img:
            aug_params = data_augmentation.sample_params(x)
            aug_img = data_augmentation(x, aug_params)
            batch.append(aug_img.cuda(non_blocking=True))
        batch = torch.cat(batch, dim=0)

        # Forward augmented images
        ft = model(batch)

        if params.angle is None:
            loss_ft = - torch.sum((ft - ft_orig) * direction)
            loss_ft_l2 = params.lambda_ft_l2 * torch.norm(ft - ft_orig, dim=1).sum()
        else:
            dot_product = torch.sum((ft - ft_orig) * direction)
            print("Dot product: ", dot_product.item())
            if params.half_cone:
                loss_ft = - rho * dot_product * torch.abs(dot_product)
            else:
                loss_ft = - rho * (dot_product ** 2)
            loss_ft_l2 = torch.norm(ft - ft_orig)**2

        loss_norm = 0
        for i in range(len(img)):
            loss_norm += params.lambda_l2_img * torch.norm(img[i].cuda(non_blocking=True) - img_orig[i].cuda(non_blocking=True))**2
        loss = loss_ft + loss_norm + loss_ft_l2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logs = {
            "keyword": "iteration",
            "loss": loss.item(),
            "loss_ft": loss_ft.item(),
            "loss_norm": loss_norm.item(),
            "loss_ft_l2": loss_ft_l2.item(),
        }
        if params.angle is not None:
            logs["R"] = - (loss_ft + loss_ft_l2).item()
        if schedule is not None:
            logs["lr"] = schedule[iteration]
        logger.info("__log__:%s" % json.dumps(logs))

        for i in range(len(img)):
            img[i].data[0] = project_linf(img[i].data[0], img_orig[i][0], params.radius)
            if iteration % 10 == 0:
                img[i].data[0] = roundPixel(img[i].data[0])

    img_new = [numpyPixel(x.data[0]).astype(np.float32) for x in img]
    img_old = [numpyPixel(x[0]).astype(np.float32) for x in img_orig]

    img_totest = torch.cat([center_da(x, 0).cuda(non_blocking=True) for x in img])
    with torch.no_grad():
        ft_new = model(img_totest)

    logger.info("__log__:%s" % json.dumps({
        "keyword": "final",
        "psnr": np.mean([psnr(x_new - x_old) for x_new, x_old in zip(img_new, img_old)]),
        "ft_direction": torch.mv(ft_new - ft_orig, direction[0]).mean().item(),
        "ft_norm": torch.norm(ft_new - ft_orig, dim=1).mean().item(),
        "rho": rho,
        "R": (rho * torch.dot(ft_new[0] - ft_orig[0], direction[0])**2 - torch.norm(ft_new - ft_orig)**2).item(),
    }))

    for i in range(len(img)):
        img_name = basename(params.img_paths[i])

        extension = ".%s" % (img_name.split(".")[-1])
        np.save(join(params.dump_path, img_name).replace(extension, ".npy"), img_new[i].astype(np.uint8))


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # debug mode
    if params.debug is True:
        params.exp_name = 'debug'
        params.debug_slurm = True
        params.debug_train = True

    # run experiment
    main(params)
