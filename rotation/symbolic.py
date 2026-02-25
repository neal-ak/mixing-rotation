"""Symbolically computes the mixing equation."""

# %%
import sympy as sym

# %%
def _preprocess_symbol(s):
    s = str(s).replace("\\", "")
    s = s.replace("^0", "").replace("_0", "").replace("^prime", "")
    s = s.removesuffix("b")
    s = s.replace("tilde{", "").replace("}", "")
    return s.strip().replace(" ", "-")

# %%
ANGLE = {
    _preprocess_symbol(k): sym.Symbol(k, real=True)
    for k in [r"\alpha", r"\beta", r"\gamma"]
}
PROP = {
    _preprocess_symbol(k): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
    for k in [r"\pi^0", r"\phi^\prime", r"\eta"]
}
BARE = {
    _preprocess_symbol(k): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
    for k in [r"\pi^0 b", r"\phi", r"\eta b", r"\pi^0 \tilde{\phi}", r"\eta \tilde{\phi}"]
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
    return rotate_gamma * rotate_beta * rotate_alpha, (rotate_alpha, rotate_beta, rotate_gamma), basic_2d


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
            [BARE["pi"] ** 2, BARE["pi-phi"]/2, 0],
            [BARE["pi-phi"]/2, BARE["phi"] ** 2, BARE["eta-phi"]/2],
            [0, BARE["eta-phi"]/2, BARE["eta"] ** 2],
        ]
    )
    mass_matrix = sym.Matrix(
        [[PROP["pi"] ** 2, 0, 0], [0, PROP["phi"] ** 2, 0], [0, 0, PROP["eta"] ** 2]]
    )
    mixing_equation = sym.Eq(
        mass_matrix,
        sym.MatMul(rotation_inverse, mixing_matrix, rotation),
        evaluate=False,
    )
    return mixing_equation, (rotation_inverse, mixing_matrix, mass_matrix)
