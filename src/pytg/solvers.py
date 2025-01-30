# Solvers for various normal mode equations such as Rayleigh and Taylor-Goldstein.

import numpy as np
from functools import wraps
from . import finite_difference as fd

# from scipy.sparse import eye
# from scipy.sparse.linalg import eigs
from scipy.linalg import eig


def check_and_reshape(arr, position):
    arr = np.asarray(arr)
    ndim = arr.ndim

    if np.isnan(arr).any():
        raise ValueError(f"Input argument at position {position} contains NaNs")

    # Ensure arr is a 1D array or a column/row vector
    if (ndim > 2) or (ndim == 2 and not any(dim == 1 for dim in arr.shape)):
        raise ValueError(
            f"Input argument at position {position} must be a 1D array or column/row vector"
        )

    # Ensure arr is a column vector
    if ndim == 2 and arr.shape[1] != 1:
        arr = arr.T

    # Ensure arr has shape (n, 1)
    if ndim == 1:
        arr = arr[:, None]

    return arr


def check_and_reshape_args(n):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            reshaped_args = [
                check_and_reshape(arg, i + 1) if i < n else arg
                for i, arg in enumerate(args)
            ]

            # check size
            if any(reshaped_args[0].size != arg.size for arg in reshaped_args[1:n]):
                raise ValueError("All input arguments must have the same size")

            return func(*reshaped_args, **kwargs)

        return wrapper

    return decorator


def ensure_equidistant_grid(z):
    z = np.squeeze(z)
    dz = z[1] - z[0]
    if not np.allclose(np.diff(z), dz, rtol=1e-5, atol=1e-8):
        raise ValueError("z must be equidistant")
    return dz


@check_and_reshape_args(2)
def rayleigh(z, U, k, acc=4):
    dz = ensure_equidistant_grid(z)
    n = len(z)

    # Differentiate U
    d2dz2 = fd.derivative_matrix(n, 2, acc, dz)
    d2Udz2 = d2dz2 @ U

    # Matrix to differentiate W
    DW2 = fd.derivative_matrix(n, 2, acc, dz, "fixed", "fixed")

    # Solve the eigenvalue problem
    I = np.eye(n)  # noqa: E741
    A = DW2 - k**2 * I
    B = -1j * k * U * A + 1j * k * d2Udz2 * I
    om, vec = eig(B, A)

    # If using sparse:
    # om, vec = eigs(B, n_modes, M=A, **eigs_kwargs)

    isort = np.argsort(-om.imag)  # equivalent to sorting by cp
    om = om[isort]
    w = vec[:, isort].real

    cp = -om.imag / k  # Phase speed
    gr = om.real  # Growth rate
    freq = om.imag  # Frequency

    return freq, gr, cp, w
