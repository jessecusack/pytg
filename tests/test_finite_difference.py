import numpy as np
from sympy import IndexedBase

import pytg.finite_difference as fd


def test_num_coefs():
    assert fd.num_coefs(deriv=2, acc=2) == (3, 4)
    assert fd.num_coefs(deriv=3, acc=2) == (5, 5)
    assert fd.num_coefs(deriv=4, acc=2) == (5, 6)
    assert fd.num_coefs(deriv=5, acc=2) == (7, 7)


def test_build_matrix_center():
    # Test order 2, accuracy 2
    mat = fd.build_matrix_center(np.zeros((5, 5)), order=2, acc=2)
    mat_o2_a2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 0, 0],
        ]
    )

    assert np.allclose(mat, mat_o2_a2)

    # Test order 1, accuracy 4
    mat = fd.build_matrix_center(np.zeros((5, 5)), order=1, acc=4)
    mat_o1_a4 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    assert np.allclose(mat, mat_o1_a4)


def test_build_exp():
    w = IndexedBase("w")
    coef = [1, 2, 3]
    exp = coef[0] * w[0] + coef[1] * w[1] + coef[2] * w[2]
    assert exp == fd.build_expr(coef)


def test_bc_none():
    # Test order 2, accuracy 2
    mat = fd.bc_none(np.zeros((5, 5)), order=2, acc=2, boundary="top")
    mat_o2_a2 = np.array(
        [
            [2.0, -5.0, 4.0, -1.0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    assert np.allclose(mat, mat_o2_a2)
    # Bottom order 2, accuracy 2
    mat = fd.bc_none(np.zeros((5, 5)), order=2, acc=2, boundary="bot")
    assert np.allclose(mat, mat_o2_a2[::-1, ::-1])

    # Test order 1, accuracy 4
    mat = fd.bc_none(np.zeros((5, 5)), order=1, acc=4, boundary="top")
    mat_o1_a4 = np.array(
        [
            [-25 / 12, 4, -3, 4 / 3, -1 / 4],
            [-1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    assert np.allclose(mat, mat_o1_a4)
    # Bottom order 1, accuracy 4
    mat = fd.bc_none(np.zeros((5, 5)), order=1, acc=4, boundary="bot")
    assert np.allclose(mat, -mat_o1_a4[::-1, ::-1])


def test_bc_fixed():
    # Test order 2, accuracy 2
    mat = fd.bc_fixed(np.zeros((5, 5)), order=2, acc=2, boundary="top")
    mat_o2_a2 = np.array(
        [
            [-2, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    assert np.allclose(mat, mat_o2_a2)
    # Bottom order 2, accuracy 2
    mat = fd.bc_fixed(np.zeros((5, 5)), order=2, acc=2, boundary="bot")
    assert np.allclose(mat, mat_o2_a2[::-1, ::-1])

    # Test order 1, accuracy 4
    mat = fd.bc_fixed(np.zeros((5, 5)), order=1, acc=4, boundary="top")
    mat_o1_a4 = np.array(
        [
            [0, 2 / 3, -1 / 12, 0, 0],
            [-2 / 3, 0, 2 / 3, -1 / 12, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    assert np.allclose(mat, mat_o1_a4)
    # Bottom order 1, accuracy 4
    mat = fd.bc_fixed(np.zeros((5, 5)), order=1, acc=4, boundary="bot")
    assert np.allclose(mat, -mat_o1_a4[::-1, ::-1])


def test_bc_zero_derivative():
    mat = fd.bc_zero_derivative(np.zeros((3, 3)), 2, 2, 1, "top")
    mat_o2_a1_d1 = np.array(
        [
            [-2 / 3, 2 / 3, 0],  # Smyth book page 144
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    assert np.allclose(mat, mat_o2_a1_d1)

    mat = fd.bc_zero_derivative(np.zeros((3, 3)), 2, 2, 1, "bot")
    assert np.allclose(mat, -mat_o2_a1_d1[::-1, ::-1])


def test_derivative_matrix():
    n = 100
    acc = 2

    # Test derivatives of sin
    z, dz = np.linspace(0, np.pi * 2, n, retstep=True)
    ddz = fd.derivative_matrix(n, 1, acc, dz)
    d2dz2 = fd.derivative_matrix(n, 2, acc, dz)
    d3dz3 = fd.derivative_matrix(n, 3, acc, dz)

    assert np.std(ddz @ np.sin(z) - np.cos(z)) < dz**acc
    assert np.std(d2dz2 @ np.sin(z) + np.sin(z)) < dz**acc
    assert np.std(d3dz3 @ np.sin(z) + np.cos(z)) < dz**acc

    # Test derivatives of polynomials
    z, dz = np.linspace(-1, 1, n, retstep=True)
    ddz = fd.derivative_matrix(n, 1, acc, dz)
    d2dz2 = fd.derivative_matrix(n, 2, acc, dz)

    assert np.std(ddz @ z**2 - 2 * z) < dz**acc
    assert np.std(d2dz2 @ z**3 - 6 * z) < dz**acc
