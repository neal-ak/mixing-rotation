"""Symbolically computes the mixing equation."""

# %%
import sympy as sym

# %%
ANGLE = {
    str(k).removeprefix("\\"): sym.Symbol(k, real=True)
    for k in [r"\alpha", r"\beta", r"\gamma"]
}
MASS = {
    str(k).removeprefix("\\"): sym.Symbol(f"M_{{{k}}}", nonnegative=True)
    for k in [r"\pi", r"\phi", r"\eta"]
}
MIX = {
    str(k).removeprefix("\\"): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
    for k in [r"\pi", r"\phi", r"\eta", "1x1", "1x2"]
}


# %%
def create_3d_rotation() -> (
    tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix], sym.Matrix]
):
    """Creates the 3D rotation matrix from three 2D rotation matrices."""
    theta = sym.Symbol(r"\theta", real=True)
    basic_2d = sym.Matrix(
        [
            [sym.cos(theta), sym.sin(theta)],
            [-sym.sin(theta), sym.cos(theta)],
        ]
    )
    rotate_alpha = sym.Matrix(sym.BlockDiagMatrix(basic_2d, sym.Matrix([1]))).subs(
        {
            sym.cos(theta): sym.cos(ANGLE["alpha"]),
            sym.sin(theta): sym.sin(ANGLE["alpha"]),
        }
    )
    rotate_beta = sym.Matrix(
        [[basic_2d[0], 0, basic_2d[1]], [0, 1, 0], [basic_2d[2], 0, basic_2d[3]]]
    ).subs(
        {
            sym.cos(theta): sym.cos(ANGLE["beta"]),
            sym.sin(theta): sym.sin(ANGLE["beta"]),
        }
    )
    rotate_gamma = sym.Matrix(
        sym.BlockDiagMatrix(
            sym.Matrix([1]),
            basic_2d.subs(
                {
                    sym.cos(theta): sym.cos(ANGLE["gamma"]),
                    sym.sin(theta): sym.sin(ANGLE["gamma"]),
                }
            ),
        )
    )
    return rotate_gamma * rotate_beta * rotate_gamma, (rotate_alpha, rotate_beta, rotate_gamma), basic_2d


# %%
def create_mix_equation(
    rotation: sym.Matrix,
) -> tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix]]:
    """
    Creates the mixing equation from the rotation, mass, and mixing matrices.
    Returns the equation, and a tuple of the matrices: (rotation_inverse,
    mixing, mass).
    """
    rotation_inverse = rotation.transpose()
    mixing_matrix = sym.Matrix(
        [
            [MIX["pi"] ** 2, MIX["1x1"], 0],
            [MIX["1x1"], MIX["phi"] ** 2, MIX["1x2"]],
            [0, MIX["1x2"], MIX["eta"] ** 2],
        ]
    )
    mass_matrix = sym.Matrix(
        [[MASS["pi"] ** 2, 0, 0], [0, MASS["phi"] ** 2, 0], [0, 0, MASS["eta"] ** 2]]
    )
    mixing_equation = sym.Eq(
        mass_matrix,
        sym.MatMul(rotation_inverse, mixing_matrix, rotation),
        evaluate=False,
    )
    return mixing_equation, (rotation_inverse, mixing_matrix, mass_matrix)
