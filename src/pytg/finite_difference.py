# Finite difference and boundary conditions generators

import math
from findiff import coefficients
from sympy import solve, collect, IndexedBase
from scipy import sparse


def num_coefs(deriv, acc):
    """Determine the number of finite difference coefficients needed for a given scheme accuracy and derivative.
    Taken from the findiff package."""
    num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    if deriv % 2 == 0:
        num_forward = num_central + 1
    else:
        num_forward = num_central
    return num_central, num_forward


def build_matrix_center(mat, order, acc):
    """Input central diagonal coefficients into a matrix."""
    coeff_dict = coefficients(order, acc)["center"]
    offsets = coeff_dict["offsets"]
    coefs = coeff_dict["coefficients"]
    nside = len(offsets) // 2
    n = mat.shape[0]
    for i in range(nside, n - nside):
        mat[i, i + offsets] = coefs
    return mat


def build_expr(coef, symbol=IndexedBase("w")):
    """Build a symbolic expression from a list of symbolic coefficients."""
    exp = 0
    for i, a in enumerate(coef):
        exp += a * symbol[i]
    return exp


def bc_none(mat, order, acc, boundary="top"):
    nc, nf = num_coefs(order, acc)
    nside = nc // 2
    if boundary == "top":
        for i in range(nside):
            mat[i, :nf] = coefficients(order, offsets=[j for j in range(-i, -i + nf)])[
                "coefficients"
            ]
    elif boundary == "bot":
        for i in range(nside):
            mat[-i - 1, -nf:] = coefficients(
                order, offsets=[j for j in range(i + 1 - nf, i + 1)]
            )["coefficients"]
    else:
        raise ValueError("Boundary condition must be either 'top' or 'bot'.")

    return mat


def bc_fixed(mat, order, acc, boundary="top"):
    # This function generates
    n, _ = num_coefs(order, acc)
    nside = n // 2
    if boundary == "top":
        for i in range(nside):
            mat[i, : n - 1] = coefficients(
                order, offsets=[j for j in range(-i - 1, n - i - 1)]
            )["coefficients"][1:]
    elif boundary == "bot":
        for i in range(nside):
            mat[-i - 1, 1 - n :] = coefficients(
                order, offsets=[j for j in range(i + 2 - n, i + 2)]
            )["coefficients"][:-1]
    else:
        raise ValueError("Boundary condition must be either 'top' or 'bot'.")
    return mat


def bc_zero_derivative(mat, order, acc, deriv, boundary):
    """The derivative (e.g. first, second, etc.) is zero at the boundary.

    mat: sparse matrix
        The finite difference matrix to apply the boundary condition to.
    order: int
        The order of the finite difference matrix.
    acc: int
        The accuracy of the finite difference scheme.
    deriv: int
        The order of the derivative to set to zero at the boundary.
    boundary: str
        The boundary to apply the condition to. Either "top" or "bot".
    """

    if deriv == order:
        return mat

    if deriv > order:
        raise ValueError(
            "The derivative order must be less than or equal to the order of the finite difference scheme."
        )

    nc, _ = num_coefs(order, acc)
    nside = nc // 2
    w = IndexedBase("w")

    if boundary == "top":
        # First we find an expression for the first derivative at the boundary
        exp = build_expr(
            coefficients(deriv, offsets=list(range(nc)), symbolic=True)["coefficients"],
            w,
        )
        # Solve for the derivative at the boundary
        dw_dz = solve(exp, w[0])[0]

        for i in range(nside):
            # Upper / left boundary
            # Offsets are shifted by 1 to include the virtual boundary points
            offsets = [j for j in range(-i - 1, nc - i - 1)]

            # Define the expression for the boundary from the coefficients.
            exp = build_expr(
                coefficients(deriv, offsets=offsets, symbolic=True)["coefficients"], w
            )

            # Substitute the value of the virtual boundary point
            sub_exp = exp.subs(w[0], dw_dz)
            collected_coefs = collect(
                sub_exp, [w[j] for j in range(1, nc)], evaluate=False
            )
            coefs = [float(collected_coefs[w[j]]) for j in range(1, nc)]

            mat[i, : nc - 1] = coefs

    elif boundary == "bot":
        # This one is confusing, index could be -4, -3 ... 0, but we flip and treat as 0, 1, ... 4.
        # First we find an expression for the first derivative at the boundary
        exp = build_expr(
            coefficients(deriv, offsets=list(range(1 - nc, 1)), symbolic=True)[
                "coefficients"
            ][::-1],
            w,
        )
        # Solve for the derivative at the boundary
        dw_dz = solve(exp, w[0])[0]

        for i in range(nside):
            # Same as for fixed
            offsets = [j for j in range(i + 2 - nc, i + 2)]

            exp = build_expr(
                coefficients(deriv, offsets=offsets, symbolic=True)["coefficients"][
                    ::-1
                ],
                w,
            )

            # Substitute the value of the virtual boundary point
            sub_exp = exp.subs(w[0], dw_dz)
            collected_coefs = collect(
                sub_exp, [w[j] for j in range(1, nc)], evaluate=False
            )
            coefs = [float(collected_coefs[w[j]]) for j in range(1, nc)]

            mat[-i - 1, -nc + 1 :] = coefs[::-1]
    else:
        raise ValueError("Boundary condition must be either 'top' or 'bot'.")
    return mat


def bc_rigid(mat, order, acc, boundary="top"):
    return bc_zero_derivative(mat, order, acc, 1, boundary)


def bc_frictionless(mat, order, acc, boundary="top"):
    return bc_zero_derivative(mat, order, acc, 2, boundary)


top_boundary = {
    "none": bc_none,
    "fixed": bc_fixed,
    "impermable": bc_fixed,
    "rigid": bc_rigid,
    "frictionless": bc_frictionless,
}


bot_boundary = {
    "none": lambda mat, order, acc: bc_none(mat, order, acc, "bot"),
    "fixed": lambda mat, order, acc: bc_fixed(mat, order, acc, "bot"),
    "impermable": lambda mat, order, acc: bc_fixed(mat, order, acc, "bot"),
    "rigid": lambda mat, order, acc: bc_rigid(mat, order, acc, "bot"),
    "frictionless": lambda mat, order, acc: bc_frictionless(mat, order, acc, "bot"),
}


def derivative_matrix(n, order, acc, dz=1.0, bc_top="none", bc_bot="none"):
    """Create a derivative matrix of size n by n.

    n: size of matrix (nxn)
    order: order of the derivative
    acc: accuracy of the scheme
    dz: spacing
    bc_top: One of "none", "fixed", "impermeable", "rigid", "frictionless".
    bc_bot: Same as for top.

    """
    mat = sparse.lil_array((n, n))
    mat = build_matrix_center(mat, order, acc)
    mat = top_boundary[bc_top](mat, order, acc)
    mat = bot_boundary[bc_bot](mat, order, acc)
    return mat / dz**order
