# @Time : 2022/2/22 21:25
# @Author : lepold
# @File : downsample_curvature.py


import numpy as np
from numpy.linalg import svd


def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def downsample_curvature(X, max_curvature, v):
    v = v * np.pi / 180  # degree to rad
    XS = X.copy()
    vx = np.array([np.sin(v[0]) * np.cos(v[1]),
                   -np.cos(v[0]) * np.cos(v[1]),
                   np.sin(v[1])]).reshape((1, -1))
    vyz = nullspace(vx)
    X = np.matmul(vyz.T, X)

    n_start = X.shape[1]
    aInd = np.arange(n_start)
    dInd = np.array([], dtype=np.int)
    seq = np.arange(n_start, dtype=np.int)

    while True:
        V = np.diff(X, 1, axis=1)
        VMag = np.sum(V[:, 1:] * V[:, :-1], axis=0) / np.sqrt(np.sum(V[:, :-1] ** 2, axis=0))
        VNorm = np.sqrt(np.sum(V[:, 1:] ** 2, axis=0) - VMag ** 2)
        VNorm = np.insert(VNorm, 0, max_curvature + 1, axis=0)
        VNorm = np.concatenate([VNorm, [max_curvature + 1]], axis=0)

        VIndL = np.where(VNorm < max_curvature)[0]
        VIndU = np.where(VNorm > max_curvature)[0]

        index = np.isin(VIndL, np.concatenate([VIndU + 1, VIndU - 1]))
        VIndL = VIndL[~index]  # 曲率小而且不是局部shape的点
        if len(VIndL) == 0:
            break
        VIndD = np.diff(VIndL)
        VIndD = np.insert(VIndD, 0, 2, axis=0)
        VIndNC = VIndL[VIndD != 1]
        VIndC = VIndL[VIndD == 1]

        VIndR = np.concatenate([VIndNC, VIndC[::2]])  # 要丢弃的点
        index = np.isin(np.arange(X.shape[1]), VIndR)
        VIndK = np.arange(X.shape[1])[~index]

        XP = X[:, VIndK]
        V = np.diff(XP, 1, axis=1)
        VMag = np.sum(V[:, 1:] * V[:, :-1], axis=0) / np.sqrt(np.sum(V[:, :-1] ** 2, axis=0))
        VNorm = np.sqrt(np.sum(V[:, 1:] ** 2, axis=0) - VMag ** 2)
        VNorm = np.insert(VNorm, 0, max_curvature, axis=0)
        VNorm = np.concatenate([VNorm, np.array([max_curvature])], axis=0)
        VIndKO = VIndK[VNorm > max_curvature]

        index = np.isin(VIndR, np.concatenate([VIndKO + 1, VIndKO - 1], axis=0))
        VIndR = VIndR[~index]

        if len(VIndR) == 0:
            break

        dInd = np.concatenate([dInd, aInd[VIndR]])
        index = np.isin(np.arange(X.shape[1]), VIndR)
        dr = np.arange(X.shape[1])[~index]
        aInd = aInd[dr]
        X = X[:, dr]
    index = np.isin(seq, dInd)
    dInd = seq[~index]
    X = XS[:, dInd]
    print(f"compression ratio: {X.shape[1] / n_start:.2f}")
    return X, dInd


def progress_bar(progress, time):
    """ Print progress bar to console output in the format
    Progress: [######### ] 90.0% in 10.22 sec

    Parameters
    ----------
    progress : float
        Value between 0 and 1.
    time : float
        Elapsed time till current progress.
    """

    print("\r|Progress: [{0:10s}] {1:.1f}% in {2:.0f} sec".format(
        '#' * int(progress * 10), progress * 100, time), end='')
    if progress >= 1:
        print("\r| Progress: [{0:10s}] {1:.1f}% in {2:.2f} sec".format(
            '#' * int(progress * 10), progress * 100, time))

    return
