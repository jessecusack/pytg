import pytg.solvers as s
import numpy as np


# Piecewise shear layer generator
def UPW(z, h, U0=1):
    U = np.zeros_like(z)
    U[z >= h] = U0
    U[(-h < z) & (z < h)] = U0 * z[(-h < z) & (z < h)] / h
    U[z <= -h] = -U0
    return U


# Analytical solution for the phase speed and growth rate for a piecewise shear layer
def f(x):
    return np.sqrt(np.exp(-4 * x) / 4 - (x - 0.5) ** 2)


def sig(k, h, U0=1):
    return U0 / h * f(k * h)


def phase_speed(k, h, U0=1):
    return 1j * sig(k, h, U0) / k


def test_equidistant_grid():
    assert np.isclose(s.ensure_equidistant_grid([1, 2, 3]), 1.0)
    assert np.isclose(s.ensure_equidistant_grid([1, 3, 5]), 2.0)
    assert np.isclose(s.ensure_equidistant_grid(np.array([1, 1.5, 2])[:, None]), 0.5)


def test_check_and_reshape():
    # Test 1D array
    arr = np.array([1, 2, 3])
    assert np.allclose(s.check_and_reshape(arr, 0), arr[:, None])

    # Test column vector
    arr = np.array([[1], [2], [3]])
    assert np.allclose(s.check_and_reshape(arr, 0), arr)

    # Test row vector
    arr = np.array([[1, 2, 3]])
    assert np.allclose(s.check_and_reshape(arr, 0), arr.T)

    # Test 2D array
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    try:
        s.check_and_reshape(arr, 0)
    except ValueError as e:
        assert (
            str(e)
            == "Input argument at position 0 must be a 1D array or column/row vector"
        )

    # Test 3D array
    arr = np.array([[[1, 2, 3], [4, 5, 6]]])
    try:
        s.check_and_reshape(arr, 0)
    except ValueError as e:
        assert (
            str(e)
            == "Input argument at position 0 must be a 1D array or column/row vector"
        )

    # Test NaNs
    arr = np.array([1, 2, np.nan])
    try:
        s.check_and_reshape(arr, 0)
    except ValueError as e:
        assert str(e) == "Input argument at position 0 contains NaNs"


def test_rayleigh():
    # Grid parameters
    n = 101
    z0 = -16.0
    z1 = 16.0
    z = np.linspace(z0, z1, n)
    acc = 4

    # Flow parameters
    U0 = 1
    h = 1.0
    k = 0.4

    _, gr, cp, _ = s.rayleigh(z, UPW(z, h, U0), k + 0 * 1j, acc=acc)
    mode_fastest = np.argmax(gr)

    assert np.isclose(cp[mode_fastest], phase_speed(k + 0 * 1j, h, U0).real, rtol=1e-3)
    assert np.isclose(gr[mode_fastest], sig(k + 0 * 1j, h, U0).real, rtol=1e-3)


def test_viscous_taylor_goldstein():
    # The Taylor-Goldstein equation should reduced to the Rayleigh equation after
    # zeroing out most of the parameters.
    # Grid parameters
    n = 101
    z0 = -16.0
    z1 = 16.0
    z = np.linspace(z0, z1, n)
    acc = 4

    # Flow parameters
    U0 = 1
    h = 1.0
    k = 0.4
    U = UPW(z, h, U0)

    _, gr, cp, _, _, _, _ = s.viscous_taylor_goldstein(
        z, U, 0 * U, 0 * U, k + 0 * 1j, 0, 0, 0, acc=acc
    )
    mode_fastest = np.argmax(gr)

    assert np.isclose(cp[mode_fastest], phase_speed(k + 0 * 1j, h, U0).real, rtol=1e-3)
    assert np.isclose(gr[mode_fastest], sig(k + 0 * 1j, h, U0).real, rtol=1e-3)
