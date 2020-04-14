# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn as nn
import argparse
from src.model import build_model
from src.stats import cosine_pvalue
from scipy.stats import combine_pvalues
import time

from src.dataset import get_data_loader
from src.net import extractFeatures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--carrier_path", type=str, required=True)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--img_size", default=256)
    parser.add_argument("--marking_network", type=str, required=True)
    parser.add_argument("--nb_workers", default=20)
    parser.add_argument("--num_classes", default=1000)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--tested_network", type=str, required=True)

    params = parser.parse_args()

    valid_data_loader, _, _ = get_data_loader(
        params,
        split='valid',
        transform='center',
        shuffle=False,
        distributed_sampler=False
    )

    carrier = torch.load(params.carrier_path).numpy()

    # Load marking network
    marking_ckpt = torch.load(params.marking_network)
    params.architecture = marking_ckpt['params']['architecture']
    print("Building %s model ..." % params.architecture)
    marking_network = build_model(params).eval().cuda()

    # Remove fully connected layer
    marking_network.fc = nn.Sequential()
    marking_state = marking_ckpt['model']
    W_old = marking_state['fc.weight'].cpu().numpy()
    del marking_state['fc.weight']
    del marking_state['fc.bias']

    print(marking_network.load_state_dict(marking_state, strict=False))

    start_all = time.time()
    tested_ckpt = torch.load(params.tested_network)
    tested_state = {k.replace("module.", ""): v for k, v in tested_ckpt['model'].items()}
    params.architecture = tested_ckpt['params']['architecture']
    print("Building %s model ..." % params.architecture)
    tested_network = build_model(params).eval().cuda()

    # Remove fully connected layer
    if params.architecture.startswith("resnet"):
        tested_network.fc = nn.Sequential()
        del tested_state['fc.weight']
        del tested_state['fc.bias']
    elif params.architecture.startswith("vgg"):
        tested_network.classifier[6] = nn.Sequential()
        del tested_state['classifier.6.weight']
        del tested_state['classifier.6.bias']
    elif params.architecture.startswith("densenet"):
        tested_network.classifier = nn.Sequential()
        del tested_state['classifier.weight']
        del tested_state['classifier.bias']

    print(tested_network.load_state_dict(tested_state, strict=False))

    # Extract features
    start = time.time()
    features_marker, _ = extractFeatures(valid_data_loader, marking_network, verbose=False)
    features_tested, _  = extractFeatures(valid_data_loader, tested_network, verbose=False)
    print("Extracting features took %.2f" % (time.time() - start))
    features_marker = features_marker.numpy()
    features_tested = features_tested.numpy()

    # Align spaces
    X, residuals, rank, s = np.linalg.lstsq(features_marker, features_tested)
    print("Norm of residual: %.4e" % np.linalg.norm(np.dot(features_marker, X) - features_tested)**2)

    # Put classification vectors into aligned space
    if params.architecture.startswith("resnet"):
        key = 'fc.weight'
        if key not in tested_ckpt['model']:
            key = 'module.fc.weight'
    elif params.architecture == "vgg16":
        key = 'classifier.6.weight'
        if key not in tested_ckpt['model']:
            key = 'module.classifier.6.weight'
    elif params.architecture.startswith("densenet"):
        key = 'classifier.weight'
        if key not in tested_ckpt['model']:
            key = 'module.classifier.weight'

    W = tested_ckpt['model'][key].cpu().numpy()
    W = np.dot(W, X.T)
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    # Computing scores
    scores = np.sum(W * carrier, axis=1)

    print("Mean p-value is at %d times sigma" % int(scores.mean() * np.sqrt(W.shape[0] * carrier.shape[1])))
    print("Epoch of the model: %d" % tested_ckpt["epoch"])

    p_vals = [cosine_pvalue(c, d=carrier.shape[1]) for c in list(scores)]
    print(f"log10(p)={np.log10(combine_pvalues(p_vals)[1])}")

    print("Total took %.2f" % (time.time() - start_all))
