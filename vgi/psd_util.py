import numpy as np
import scipy as sp


def project_psd(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_psd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_psd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def psd_sqrt(A):
    """Matrix square root of a PSD matrix (or a PSD approximation of it)"""
    L = np.linalg.cholesky(project_psd(A))
    return L.T


def low_rank_factor(A, r):
    """Rank r approximate factorization of PSD matrix A (or a PSD approximation of it)"""
    n = A.shape[0]
    assert r >= 0

    if r == 0:
        return np.zeros(A.shape)

    if r >= n:
        return psd_sqrt(A)

    if r < n - 1:
        lambdas, Q = sp.sparse.linalg.eigs(A, k=r)
        return np.sqrt(np.diag(lambdas.real)) @ Q.real.T

    lambdas, Q = np.linalg.eig(A)
    return np.diag(np.sqrt(np.clip(lambdas.real[:r], 0, None))) @ Q.real[:, :r].T


def is_psd(A):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
