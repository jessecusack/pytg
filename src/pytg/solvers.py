# Solvers for various normal mode equations such as Rayleigh and Taylor-Goldstein.

import numpy as np
from functools import wraps
from . import finite_difference as fd

# from scipy.sparse import eye
# from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from findiff import FinDiff


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
    if dz < 0:
        raise ValueError("z must be monotonically increasing")
    return dz


@check_and_reshape_args(2)
def rayleigh(z, U, k, acc=4):
    dz = ensure_equidistant_grid(z)
    n = len(z)

    # Differentiate U
    D2 = fd.derivative_matrix(n, 2, acc, dz)
    Uzz = D2 @ U

    # Matrix to differentiate W
    DW2 = fd.derivative_matrix(n, 2, acc, dz, "fixed", "fixed")

    # Solve the eigenvalue problem
    I = np.eye(n)  # noqa: E741
    A = DW2 - k**2 * I
    B = -1j * k * U * A + 1j * k * Uzz * I
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


@check_and_reshape_args(4)
def viscous_taylor_goldstein(
    z,
    U,
    V,
    B,
    k,
    l,
    Kv,
    Kb,
    vbc_top="rigid",
    vbc_bot="rigid",
    bbc_top="fixed",
    bbc_bot="fixed",
    acc=4,
):
    dz = ensure_equidistant_grid(z)
    n = U.size
    kh = np.sqrt(k**2 + l**2)  # Absolute horizontal wavenumber
    # Velocity component parallel to the wave vector (k, l)
    U = U * k / kh + V * l / kh

    # Derivative matrices
    D1 = fd.derivative_matrix(n, 1, acc, dz)
    D2 = fd.derivative_matrix(n, 2, acc, dz)

    # Shear and buoyancy frequency.
    Bz = D1 @ B
    Uzz = D2 @ U

    # Create matrices with boundary conditions
    DW2 = fd.derivative_matrix(n, 2, acc, dz, "fixed", "fixed")  # Impermeable
    DW4 = fd.derivative_matrix(n, 4, acc, dz, vbc_top, vbc_bot)
    DB2 = fd.derivative_matrix(n, 2, acc, dz, bbc_top, bbc_bot)

    # Assemble matrices for eigenvalue computation
    I = np.eye(n)  # noqa: E741
    L = DW2 - I * kh**2  # Laplacian
    Lb = DB2 - I * kh**2  # Laplacian for buoyancy
    LL = DW4 - 2 * DW2 * kh**2 + I * kh**4  # Laplacian of laplacian

    A = np.block([[L, np.zeros_like(L)], [np.zeros_like(L), I]])

    b11 = (
        -1j * k * np.diag(np.squeeze(U)) @ L
        + 1j * k * np.diag(np.squeeze(Uzz))
        + Kv * LL
    )
    b21 = -np.diag(np.squeeze(Bz))
    b12 = -I * kh**2
    b22 = -1j * k * np.diag(np.squeeze(U)) + Kb * Lb

    B = np.block([[b11, b12], [b21, b22]])

    om, vec = eig(B, A)

    isort = np.argsort(-om.imag)  # equivalent to sorting by cp
    om = om[isort]
    vec = vec[:, isort]
    w = vec[:n, :]
    b = vec[n:, :]

    cp = -om.imag / k  # Phase speed
    gr = om.real  # Growth rate
    freq = om.imag  # Frequency

    d_dz = FinDiff(0, dz, 1, acc=acc)
    d2_dz2 = FinDiff(0, dz, 2, acc=acc)

    Uvec = 1j * d_dz(w) / k

    theta = np.arctan2(l, k)
    u = Uvec * np.cos(theta)
    v = Uvec * np.sin(theta)

    X1 = d_dz(U) * w
    X2 = (cp - U) * d_dz(w)
    X3 = 1j * Kv * (d2_dz2(d_dz(w)) - k * d_dz(w)) / k
    p = 1j * (X1 + X2 - X3) / k

    return freq, gr, cp, w.real, b.real, u.real, v.real, p.real


@check_and_reshape_args(5)
def viscous_symmetric(
    z,
    U,
    V,
    B,
    By,
    k,
    l,
    Kv,
    Kb,
    f = 1e-4,
    vbc_top="rigid",
    vbc_bot="rigid",
    bbc_top="fixed",
    bbc_bot="fixed",
    acc=4,
):
    dz = ensure_equidistant_grid(z)
    n = U.size
    kh = np.sqrt(k**2 + l**2)  # Absolute horizontal wavenumber
    # Velocity component parallel to the wave vector (k, l)
    U = U * k / kh + V * l / kh

    # Derivative matrices
    D1 = fd.derivative_matrix(n, 1, acc, dz).toarray()
    D2 = fd.derivative_matrix(n, 2, acc, dz).toarray()
    D4 = fd.derivative_matrix(n, 4, acc, dz).toarray()

    # Shear and buoyancy frequency.
    Bz = D1 @ B
    Uz = D1 @ U

    # Create matrices with boundary conditions
    DW1 = fd.derivative_matrix(n, 1, acc, dz, "fixed", "fixed").toarray()  # Impermeable
    DQ1 = fd.derivative_matrix(n, 1, acc, dz, "rigid", "rigid").toarray()
    # DB2 = fd.derivative_matrix(n, 1, acc, dz, "insulating", "insulating")

    # Assemble matrices for eigenvalue computation
    I = np.eye(n)  # noqa: E741
    L = D2 - I * kh**2  # Laplacian
    # LB = DB2 - I * kh**2
    LL = D4 - 2 * D2 * kh**2 + I * kh**4  # Laplacian of laplacian

    A = np.block([[I, np.zeros_like(L), np.zeros_like(L)],
                  [np.zeros_like(L), L, np.zeros_like(L)], 
                  [np.zeros_like(L), np.zeros_like(L), I]])


    b11 = -1j*k*np.diag(np.squeeze(U))@L+ Kv*L
    b12 = 1j*l*Uz + f*DW1
    b13 = np.zeros_like(I)

    b21 = -f*DQ1
    b22 = (
        -1j * k * np.diag(np.squeeze(U)) @ L
        + Kv * LL
    )
    b23 = -I * kh**2

    b31 = 1j*k / kh**2 * np.diag(np.squeeze(By))
    b32 = 1j*l / kh**2 * np.diag(np.squeeze(Bz)) @ DW1 - np.diag(np.squeeze(Bz))
    b33 = -1j * k * np.diag(np.squeeze(U)) + Kb * L

    B = np.block([[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]])

    om, vec = eig(B, A)

    isort = np.argsort(-om.imag)  # equivalent to sorting by cp
    om = om[isort]
    vec = vec[:, isort]
    q = vec[:n, :]
    w = vec[n:2*n, :]
    b = vec[2*n:, :]

    cp = -om.imag / k  # Phase speed
    gr = om.real  # Growth rate
    freq = om.imag  # Frequency

    # d_dz = FinDiff(0, dz, 1, acc=acc)
    # d2_dz2 = FinDiff(0, dz, 2, acc=acc)

    # Uvec = 1j * d_dz(w) / k

    # theta = np.arctan2(l, k)
    # u = Uvec * np.cos(theta)
    # v = Uvec * np.sin(theta)

    # X1 = d_dz(U) * w
    # X2 = (cp - U) * d_dz(w)
    # X3 = 1j * Kv * (d2_dz2(d_dz(w)) - k * d_dz(w)) / k
    # p = 1j * (X1 + X2 - X3) / k

    # return freq, gr, cp, w.real, b.real, u.real, v.real, p.real
    return freq, gr, cp, q.real, w.real, b.real
