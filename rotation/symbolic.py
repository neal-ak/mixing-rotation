"""Symbolically computes the mixing equation."""

# %%
import sympy as sym

# %%
def _preprocess_symbol(s):
    s = str(s).replace("\\", "")
    s = s.replace("^prime", "").replace("^", "").replace("_", "")
    s = s.removesuffix("b")
    s = s.replace("tilde{", "").replace("}", "")
    return s.strip().replace(" ", "-")

# %%
ANGLE = {
    _preprocess_symbol(k): sym.Symbol(k, real=True)
    for k in [r"\alpha", r"\beta", r"\gamma", "u", "v", "w"]
}
TRIG = {t: {k: sym.Symbol(f"{t}_{v}", real=True) for k,v in ANGLE.items()} for t in ["c", "s", "t"]}
PROP = {
    _preprocess_symbol(k): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
    for k in [r"\pi^0", r"\tilde{\phi}^\prime", r"\eta", r"\eta_0"]
}
BARE = {
    _preprocess_symbol(k): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
    for k in [r"\pi^0 b", r"\phi", r"\eta b", r"\pi^0 \tilde{\phi}", r"\eta \tilde{\phi}", r"\eta_0"]
}


# %%
def create_3d_rotation() -> (
    tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix], sym.Matrix]
):
    """Creates the 3D rotation matrix from three 2D rotation matrices."""
    theta = sym.Symbol(r"\theta", real=True)
    basic_2d = sym.Matrix(
        [
            [sym.cos(theta), -sym.sin(theta)],
            [sym.sin(theta), sym.cos(theta)],
        ]
    )
    rotate = [
        sym.Matrix(sym.BlockDiagMatrix(basic_2d, sym.Matrix([1]))),
        sym.Matrix([[basic_2d[0], 0, basic_2d[1]], [0, 1, 0], [basic_2d[2], 0, basic_2d[3]]]),
        sym.Matrix(sym.BlockDiagMatrix(sym.Matrix([1]), basic_2d)),
    ]
    for i, (m,s) in enumerate(zip(rotate, [ANGLE[k] for k in ["alpha", "beta", "gamma"]])):
        rotate[i] = m.subs(theta, s)
    return rotate[0] * rotate[1] * rotate[2], tuple(rotate), basic_2d


# %%
def create_mix_equation(
    rotation: sym.Matrix,
) -> tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix]]:
    """
    Creates the mixing equation from the rotation, propagating, and bare matrices.
    Returns the equation, and a tuple of the matrices: (rotation_transpose,
    bare, propagating).
    """
    rotation_transpose = rotation.transpose()
    bare_matrix = sym.Matrix(
        [
            [BARE["pi0"] ** 2, BARE["pi0-phi"]/2, 0],
            [BARE["pi0-phi"]/2, BARE["phi"] ** 2, BARE["eta-phi"]/2],
            [0, BARE["eta-phi"]/2, BARE["eta"] ** 2],
        ]
    )
    prop_matrix = sym.Matrix(
        [[PROP["pi0"] ** 2, 0, 0], [0, PROP["phi"] ** 2, 0], [0, 0, PROP["eta"] ** 2]]
    )
    mixing_equation = sym.Eq(
        prop_matrix,
        sym.MatMul(rotation, bare_matrix, rotation_transpose),
        evaluate=False,
    )
    return mixing_equation, (rotation_transpose, bare_matrix, prop_matrix)

# %%
def create_kinetic():
    angles = [ANGLE[k] for k in ["u", "v", "w"]]
    d = sym.sqrt(1 - sym.sin(angles[0])**2 - sym.sin(angles[1])**2 - sym.sin(angles[2])**2 + 2*sym.sin(angles[0])*sym.sin(angles[1])*sym.sin(angles[2]))
    return sym.Matrix([
        [1, -sym.tan(angles[0]), (sym.tan(angles[0])*sym.sin(angles[2])-sym.sin(angles[1])/sym.cos(angles[0]))/d],
        [0, 1/sym.cos(angles[0]), (sym.tan(angles[0])*sym.sin(angles[1])-sym.sin(angles[2])/sym.cos(angles[0]))/d],
        [0, 0, sym.cos(angles[0])/d]
    ])

# %%
def create_axial_mix_equation(rotation, kinetic):
    rotation_transpose = rotation.transpose()
    kinetic_transpose = kinetic.transpose()
    bare_matrix = sym.Matrix(
        [
            [BARE["pi0"] ** 2, 0, 0],
            [0, BARE["eta"] ** 2, 0],
            [0, 0, BARE["eta0"] ** 2],
        ]
    )
    prop_matrix = sym.Matrix(
        [[PROP["pi0"] ** 2, 0, 0], [0, PROP["eta"] ** 2, 0], [0, 0, PROP["eta0"] ** 2]]
    )
    mixing_equation = sym.Eq(
        prop_matrix,
        sym.MatMul(rotation, kinetic, bare_matrix, kinetic_transpose, rotation_transpose),
        evaluate=False,
    )
    return mixing_equation, (kinetic_transpose, rotation_transpose, bare_matrix, prop_matrix)
