# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from scipy.special import betainc


def cosine_pvalue(c, d):
    """
    Given a dimension d, returns the probability that the dot product between
    random unitary vectors is higher than c
    """
    assert type(c) in [float, np.float64, np.float32]

    a = (d - 1) / 2.
    b = 1 / 2.

    if c >= 0:
        return 0.5 * betainc(a, b, 1-c**2)
    else:
        return 1 - cosine_pvalue(-c, d=d)



if __name__ == "__main__":
    n, d = 10000, 20

    x = np.random.randn(d)
    x /= np.linalg.norm(x)

    vecs = np.random.randn(n, d)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    cos = np.dot(vecs, x)

    for s in np.arange(-1, 1.1, 0.1):
        p_value_theoretical = cosine_pvalue(s, d=d)
        p_value_empirical = np.mean(cos >= s)
        print(f"P(cos >= {s}): {p_value_empirical} (empirical) ~= {p_value_theoretical} (theoretical)")
