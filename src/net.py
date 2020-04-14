# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# import json
import time
import torch

def extractFeatures(loader, model, ignore_first=False, numpy=False, verbose=True):
    assert not model.training
    with torch.no_grad():
        n = len(loader.dataset)
        features = None
        # targets = np.zeros((n), dtype=int)

        offset = 0
        start = time.time()
        for elements in loader:
            if ignore_first:
                elements = elements[1:]

            img = elements[0]
            ft = model(img.cuda(non_blocking=True))
            sz = img.size(0)

            if features is None:
                d = ft.size(1)
                # features = np.zeros((n, d), dtype=np.float32)
                features = torch.zeros((n, d), dtype=torch.float32)
                all_targets = [torch.zeros((n, ), dtype=int) for _ in elements[1:]]

            features[offset:offset+sz] = ft.cpu().detach()#.numpy()

            for target, targets in zip(elements[1:], all_targets):
                targets[offset:offset+sz] = target#.numpy()

            offset += sz
            if offset % (100 * sz) == 0 and verbose:
                speed = offset / (time.time() - start)
                eta = (len(loader)*sz - offset) / speed
                print(f"Speed: {speed}, ETA: {eta}")

            # if offset >= 20000 and n >= 1e6:
            #     for i in range(10):
            #         print(20 * "*")
            #     print("WARNING: Quit extractFeatures before having extracted features of all samples")
            #     for i in range(10):
            #         print(20 * "*")
            #     break

        assert offset == n
        # if offset < n:
        #     features = features[:offset]
        #     all_targets = tuple([targets[:offset] for targets in all_targets])

        if numpy:
            features = features.numpy()
            all_targets = [targets.numpy() for targets in all_targets]

        return (features,) + tuple(all_targets)
