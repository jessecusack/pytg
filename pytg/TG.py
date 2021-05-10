import findiff as fd
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import eig


def vTG(
    z,
    u,
    v,
    b,
    k,
    l,
    Kv,
    Kb,
    BCv_upper="rigid",
    BCv_lower="rigid",
    BCb_upper="constant",
    BCb_lower="constant",
):
    """
    Solver for the viscous Taylor Goldstein equation for the case of constant viscous/diffusive coefficients.

    Parameters
    ----------
        z : array
            Height [m]. Must be equally spaced.
        u : array
            Zonal velocity [m s-1].
        v : array
            Meridional velocity [m s-1].
        b : array
            Buoyancy [m s-2].
        k : float
            Zonal wavenumber (angular) [rad m-1].
        l : float
            Meridional wavenumber (angular) [rad m-1].
        Kv : float
            Momentum diffusivity or viscosity [m2 s-1].
        Kv : float
            Buoyancy diffusivity [m2 s-1].
        BCv_upper : string
            Upper boundary condition on velocity, either "rigid" (default) or "frictionless".
        BCv_lower : string
            Lower boundary condition on velocity, either "rigid" (default) or "frictionless".
        BCb_upper : string
            Upper boundary condition on buoyancy, either "constant" (default) or "insulating".
        BCb_lower : string
            Lower boundary condition on buoyancy, either "constant" (default) or "insulating".

    Returns
    -------
        om : array
            Complex frequency where the real part is the growth rate and imaginary part is the frequency (angular) [rad s-1].
        wvec : 2d array
            Eigenvectors of vertical velocity.
        bvec : 2d array
            Eigenvectors of buoyancy.

    """

    z = np.asarray(z)
    u = np.asarray(u)
    v = np.asarray(v)
    b = np.asarray(b)

    if not (z.size == u.size == v.size == b.size):
        raise ValueError(
            "Size of z, u, v and b must be equal, z.size = {}, u.size = {}, v.size = {}, b.size = {}.".format(
                z.size, u.size, v.size, b.size
            )
        )

    dz = z[1] - z[0]
    # check for equally spaced z
    if not np.all(np.diff(z) == dz):
        raise ValueError("z values are not equally spaced.")

    flip_data = False
    if dz < 0:
        flip_data = True
        dz *= -1
        u = np.flipud(u)
        v = np.flipud(v)
        b = np.flipud(b)

    N = u.size
    kh = np.sqrt(k ** 2 + l ** 2)  # Absolute horizontal wavenumber

    # Velocity component parallel to the wave vector (k, l)
    U = u * k / kh + v * l / kh

    # Derivative matrices
    # 1st derivative
    dz1 = fd.FinDiff(0, dz, 1).matrix(u.shape).toarray()
    # 2nd derivative
    dz2 = fd.FinDiff(0, dz, 2).matrix(u.shape).toarray()
    # 4th derivative
    dz4 = fd.FinDiff(0, dz, 4).matrix(u.shape).toarray()

    # Shear and buoyancy frequency.
    bz = dz1 @ b
    Uzz = dz2 @ U

    # Add boundary conditions to matrix
    # Impermeable boundary
    dz2[0, :] = 0
    dz2[0, 0] = -2 / dz ** 2
    dz2[0, 1] = 1 / dz ** 2
    dz2[-1, :] = 0
    dz2[-1, -1] = -2 / dz ** 2
    dz2[-1, -2] = 1 / dz ** 2

    # % Asymptotic boundary
    # % D2(1,:)=0;
    # % D2(1,1)=2*(-del*kt-1)/del^2;
    # % D2(1,2)=2/del^2;
    # % D2(N,:)=0;
    # % D2(N,N)=2*(-del*kt-1)/del^2;
    # % D2(N,N-1)=2/del^2;

    if BCv_upper == "rigid":
        BCv1 = 0
    elif BCv_upper == "frictionless":
        BCv1 = 1
    else:
        raise ValueError(
            "BCv_upper incorrectly specified, it must be either 'rigid' or 'frictionless'."
        )

    if BCv_lower == "rigid":
        BCvN = 0
    elif BCv_lower == "frictionless":
        BCvN = 1
    else:
        raise ValueError(
            "BCv_lower incorrectly specified, it must be either 'rigid' or 'frictionless'."
        )

    # % Rigid or frictionless BCs for 4th derivative
    dz4[0, :] = 0
    dz4[0, 0] = (5 + 2 * BCv1) / dz ** 4
    dz4[0, 1] = -4 / dz ** 4
    dz4[0, 2] = 1 / dz ** 4
    dz4[1, :] = 0
    dz4[1, 0] = -4 / dz ** 4
    dz4[1, 1] = 6 / dz ** 4
    dz4[1, 2] = -4 / dz ** 4
    dz4[1, 3] = 1 / dz ** 4
    dz4[-1, :] = 0
    dz4[-1, -1] = (5 + 2 * BCvN) / dz ** 4
    dz4[-1, -2] = -4 / dz ** 4
    dz4[-1, -3] = 1 / dz ** 4
    dz4[-2, :] = 0
    dz4[-2, -1] = -4 / dz ** 4
    dz4[-2, -2] = 6 / dz ** 4
    dz4[-2, -3] = -4 / dz ** 4
    dz4[-2, -4] = 1 / dz ** 4

    # Boundary conditions for the second derivative of buoyancy.
    dz2b = dz2.copy()

    if BCb_upper == "constant":
        dz2b[0, :] = 0
        dz2b[0, 0] = -2 / dz ** 2
        dz2b[0, 1] = 1 / dz ** 2
    elif BCb_upper == "insulating":
        dz2b[0, :] = 0
        dz2b[0, 0] = -2 / (3 * dz ** 2)
        dz2b[0, 1] = 2 / (3 * dz ** 2)
    else:
        raise ValueError(
            "BCb_upper incorrectly specified, it must be either 'constant' or 'insulating'."
        )

    if BCb_lower == "constant":
        dz2b[-1, :] = 0
        dz2b[-1, -1] = -2 / dz ** 2
        dz2b[-1, -2] = 1 / dz ** 2
    elif BCb_lower == "insulating":
        dz2b[-1, :] = 0
        dz2b[-1, -1] = -2 / (3 * dz ** 2)
        dz2b[-1, -2] = 2 / (3 * dz ** 2)
    else:
        raise ValueError(
            "BCb_lower incorrectly specified, it must be either 'constant' or 'insulating'."
        )

    # Assemble stability matrices for eigenvalue computation
    Id = np.eye(N)
    L = dz2 - Id * kh ** 2  # Laplacian
    Lb = dz2b - Id * kh ** 2  # Laplacian for buoyancy
    LL = dz4 - 2 * dz2 * kh ** 2 + Id * kh ** 4  # Laplacian of laplacian

    A = np.block([[L, np.zeros_like(L)], [np.zeros_like(L), Id]])

    b11 = -1j * k * np.diag(U) @ L + 1j * k * np.diag(Uzz) + Kv * LL
    b21 = -np.diag(bz)
    b12 = -Id * kh ** 2
    b22 = -1j * k * np.diag(U) + Kb * Lb

    B = np.block([[b11, b12], [b21, b22]])

    # Solve system using eig which returns om the imaginary frequency and vec the eigenvectors.
    om, vec = eig(B, A)

    # Prepare output
    # Sort output by phase speed
    cp = -om.imag / kh
    idxs = np.argsort(cp)

    cp = cp[idxs]
    om = om[idxs]
    vec = vec[:, idxs]

    wvec = vec[:N, :]
    bvec = vec[N:, :]

    d_dz = fd.FinDiff(0, dz, 1, acc=5)
    uvec = 1j * d_dz(wvec) / k
    
    d2_dz2 = fd.FinDiff(0, dz, 2, acc=5)
    X1 = d_dz(U)[:, np.newaxis] * wvec
    X2 = (cp[np.newaxis, :] - U[:, np.newaxis]) * d_dz(wvec)
    X3 = 1j * Kv * (d2_dz2(d_dz(wvec)) - k*d_dz(wvec)) / k
    pvec = 1j * (X1 + X2 - X3) / k

    if flip_data:
        wvec = np.flipud(wvec)
        bvec = np.flipud(bvec)
        uvec = np.flipud(uvec)
        pvec = np.flipud(pvec)

    return om, wvec, bvec, uvec, pvec


def vTG_sparse(
    z,
    u,
    v,
    b,
    k,
    l,
    Kv,
    Kb,
    BCv_upper="rigid",
    BCv_lower="rigid",
    BCb_upper="constant",
    BCb_lower="constant",
    nmodes=10,
    which="LM",
):
    """
    Solver for the viscous Taylor Goldstein equation for the case of constant viscous/diffusive coefficients using sparse matrix algorithms.

    Parameters
    ----------
        z : array
            Height [m]. Must be equally spaced.
        u : array
            Zonal velocity [m s-1].
        v : array
            Meridional velocity [m s-1].
        b : array
            Buoyancy [m s-2].
        k : float
            Zonal wavenumber (angular) [rad m-1].
        l : float
            Meridional wavenumber (angular) [rad m-1].
        Kv : float
            Momentum diffusivity or viscosity [m2 s-1].
        Kv : float
            Buoyancy diffusivity [m2 s-1].
        BCv_upper : string
            Upper boundary condition on velocity, either "rigid" (default) or "frictionless".
        BCv_lower : string
            Lower boundary condition on velocity, either "rigid" (default) or "frictionless".
        BCb_upper : string
            Upper boundary condition on buoyancy, either "constant" (default) or "insulating".
        BCb_lower : string
            Lower boundary condition on buoyancy, either "constant" (default) or "insulating".
        nmodes : int
            Number of modes to estimate (will return 2*nmodes)
        which : string
            sparse.linalg.eigs argument which

    Returns
    -------
        om : array
            Complex frequency where the real part is the growth rate and imaginary part is the frequency (angular) [rad s-1].
        wvec : 2d array
            Eigenvectors of vertical velocity.
        bvec : 2d array
            Eigenvectors of buoyancy.

    """

    z = np.asarray(z)
    u = np.asarray(u)
    v = np.asarray(v)
    b = np.asarray(b)

    if not (z.size == u.size == v.size == b.size):
        raise ValueError(
            "Size of z, u, v and b must be equal, z.size = {}, u.size = {}, v.size = {}, b.size = {}.".format(
                z.size, u.size, v.size, b.size
            )
        )

    dz = z[1] - z[0]
    # check for equally spaced z
    if not np.all(np.diff(z) == dz):
        raise ValueError("z values are not equally spaced.")

    flip_data = False
    if dz < 0:
        flip_data = True
        dz *= -1
        u = np.flipud(u)
        v = np.flipud(v)
        b = np.flipud(b)

    N = u.size
    kh = np.sqrt(k ** 2 + l ** 2)  # Absolute horizontal wavenumber

    # Velocity component parallel to the wave vector (k, l)
    U = u * k / kh + v * l / kh

    # Derivative matrices
    # 1st derivative
    dz1 = fd.FinDiff(0, dz, 1).matrix(u.shape).toarray()
    # 2nd derivative
    dz2 = fd.FinDiff(0, dz, 2).matrix(u.shape).toarray()
    # 4th derivative
    dz4 = fd.FinDiff(0, dz, 4).matrix(u.shape).toarray()

    # Shear and buoyancy frequency.
    bz = dz1 @ b
    Uzz = dz2 @ U

    # Add boundary conditions to matrix
    # Impermeable boundary
    dz2[0, :] = 0
    dz2[0, 0] = -2 / dz ** 2
    dz2[0, 1] = 1 / dz ** 2
    dz2[-1, :] = 0
    dz2[-1, -1] = -2 / dz ** 2
    dz2[-1, -2] = 1 / dz ** 2

    if BCv_upper == "rigid":
        BCv1 = 0
    elif BCv_upper == "frictionless":
        BCv1 = 1
    else:
        raise ValueError(
            "BCv_upper incorrectly specified, it must be either 'rigid' or 'frictionless'."
        )

    if BCv_lower == "rigid":
        BCvN = 0
    elif BCv_lower == "frictionless":
        BCvN = 1
    else:
        raise ValueError(
            "BCv_lower incorrectly specified, it must be either 'rigid' or 'frictionless'."
        )

    # % Rigid or frictionless BCs for 4th derivative
    dz4[0, :] = 0
    dz4[0, 0] = (5 + 2 * BCv1) / dz ** 4
    dz4[0, 1] = -4 / dz ** 4
    dz4[0, 2] = 1 / dz ** 4
    dz4[1, :] = 0
    dz4[1, 0] = -4 / dz ** 4
    dz4[1, 1] = 6 / dz ** 4
    dz4[1, 2] = -4 / dz ** 4
    dz4[1, 3] = 1 / dz ** 4
    dz4[-1, :] = 0
    dz4[-1, -1] = (5 + 2 * BCvN) / dz ** 4
    dz4[-1, -2] = -4 / dz ** 4
    dz4[-1, -3] = 1 / dz ** 4
    dz4[-2, :] = 0
    dz4[-2, -1] = -4 / dz ** 4
    dz4[-2, -2] = 6 / dz ** 4
    dz4[-2, -3] = -4 / dz ** 4
    dz4[-2, -4] = 1 / dz ** 4

    # Boundary conditions for the second derivative of buoyancy.
    dz2b = dz2.copy()

    if BCb_upper == "constant":
        dz2b[0, :] = 0
        dz2b[0, 0] = -2 / dz ** 2
        dz2b[0, 1] = 1 / dz ** 2
    elif BCb_upper == "insulating":
        dz2b[0, :] = 0
        dz2b[0, 0] = -2 / (3 * dz ** 2)
        dz2b[0, 1] = 2 / (3 * dz ** 2)
    else:
        raise ValueError(
            "BCb_upper incorrectly specified, it must be either 'constant' or 'insulating'."
        )

    if BCb_lower == "constant":
        dz2b[-1, :] = 0
        dz2b[-1, -1] = -2 / dz ** 2
        dz2b[-1, -2] = 1 / dz ** 2
    elif BCb_lower == "insulating":
        dz2b[-1, :] = 0
        dz2b[-1, -1] = -2 / (3 * dz ** 2)
        dz2b[-1, -2] = 2 / (3 * dz ** 2)
    else:
        raise ValueError(
            "BCb_lower incorrectly specified, it must be either 'constant' or 'insulating'."
        )

    # Assemble stability matrices for eigenvalue computation
    Id = np.eye(N)
    L = dz2 - Id * kh ** 2  # Laplacian
    Lb = dz2b - Id * kh ** 2  # Laplacian for buoyancy
    LL = dz4 - 2 * dz2 * kh ** 2 + Id * kh ** 4  # Laplacian of laplacian

    A = np.block([[L, np.zeros_like(L)], [np.zeros_like(L), Id]])

    b11 = -1j * k * np.diag(U) @ L + 1j * k * np.diag(Uzz) + Kv * LL
    b21 = -np.diag(bz)
    b12 = -Id * kh ** 2
    b22 = -1j * k * np.diag(U) + Kb * Lb

    B = np.block([[b11, b12], [b21, b22]])

    # Solve system using eig which returns om the imaginary frequency and vec the eigenvectors.
    A_ = sparse.csc_matrix(A)
    B_ = sparse.csc_matrix(B)

    om, vec = sparse.linalg.eigs(B_, 2 * nmodes, A_, which=which)

    # Prepare output
    # Sort output by phase speed
    cp = -om.imag / kh
    idxs = np.argsort(cp)

    cp = cp[idxs]
    om = om[idxs]
    vec = vec[:, idxs]

    wvec = vec[:N, :]
    bvec = vec[N:, :]

    d_dz = fd.FinDiff(0, dz, 1, acc=5)
    uvec = 1j * d_dz(wvec) / k

    if flip_data:
        wvec = np.flipud(wvec)
        bvec = np.flipud(bvec)

    return om, wvec, bvec, uvec
