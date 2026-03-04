"""General use for symbolic calculations."""

# %%
import sympy as sym


# %%
def _preprocess_symbol(s):
    """Helper to automate assigning dict key strings from symbols."""
    s = str(s).replace("\\", "")
    s = s.replace("^prime", "").replace("^", "").replace("_", "")
    s = s.replace(" b", "")
    s = s.replace("tilde{", "").replace("}", "")
    return s.strip().replace(" ", "-")


# %%
ALL_SYMBOL = {
    "angle": {
        _preprocess_symbol(k): sym.Symbol(k, real=True)
        for k in [r"\alpha", r"\beta", r"\gamma", "u", "v", "w"]
    },
    "prop": {
        _preprocess_symbol(k): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
        for k in [
            r"\pi^0",
            r"\tilde{\phi}^\prime",
            r"\eta",
            r"\eta_0",
            r"A^\prime",
            r"K^0",
        ]
    },
    "bare": {
        _preprocess_symbol(k): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
        for k in [
            r"\pi^0 b",
            r"\phi",
            r"\eta b",
            r"\pi^0 \tilde{\phi}",
            r"\eta \tilde{\phi}",
            r"\eta_0 b",
            r"\pi^0 A",
            r"\eta A",
            r"\eta_0 A",
        ]
    },
    "coupling": {
        _preprocess_symbol(k): sym.Symbol(f"\\tilde{{g}}_{{{k}}}", real=True)
        for k in [
            "u",
            "d",
            "s",
            r"\pi b \chi",
            r"\tilde{\phi} \chi",
            r"\eta b \chi",
            r"\pi b \gamma",
            r"\tilde{\phi} \gamma",
            r"\eta b \gamma",
            r"\pi b \pi K",
            r"\tilde{\phi} \pi K",
            r"\eta b \pi K",
        ]
    },
    "rotated": {
        _preprocess_symbol(k): sym.Symbol(f"\\tilde{{g}}_{{{k}}}", real=True)
        for k in [
            r"\pi \chi",
            r"\tilde{\phi}^\prime \chi",
            r"\eta \chi",
            r"\pi \gamma",
            r"\tilde{\phi}^\prime \gamma",
            r"\eta \gamma",
            r"\pi \pi K",
            r"\tilde{\phi}^\prime \pi K",
            r"\eta \pi K",
        ]
    },
    "parameters": {_preprocess_symbol(k): sym.Symbol(k, real=True) for k in [r"E_\pi"]},
}
for t in ["cos", "sin", "tan"]:
    ALL_SYMBOL[t] = {
        k: sym.Symbol(f"{t[0]}_{v}", real=True) for k, v in ALL_SYMBOL["angle"].items()
    }


# %%
def create_3d_rotation() -> (
    tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix], sym.Matrix]
):
    """Creates the 3D rotation matrix from three 2D rotation matrices.

    :return: Rotation matrix, (alpha, beta, gamma) rotations, 2D matrix
    :rtype: tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix], sym.Matrix]
    """
    theta = sym.Symbol(r"\theta", real=True)
    basic_2d = sym.Matrix(
        [
            [sym.cos(theta), -sym.sin(theta)],
            [sym.sin(theta), sym.cos(theta)],
        ]
    )
    rotate = [
        sym.Matrix(sym.BlockDiagMatrix(basic_2d, sym.Matrix([1]))),
        sym.Matrix(
            [[basic_2d[0], 0, basic_2d[1]], [0, 1, 0], [basic_2d[2], 0, basic_2d[3]]]
        ),
        sym.Matrix(sym.BlockDiagMatrix(sym.Matrix([1]), basic_2d)),
    ]
    for i, (m, s) in enumerate(
        zip(rotate, [ALL_SYMBOL["angle"][k] for k in ["alpha", "beta", "gamma"]])
    ):
        rotate[i] = m.subs(theta, s)
    return rotate[0] * rotate[1] * rotate[2], tuple(rotate), basic_2d


# %%
def create_mix_equation(
    rotation: sym.Matrix,
) -> tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix]]:
    """Creates the mixing equation.

    :param rotation: Rotation matrix
    :type rotation: sym.Matrix
    :return: Mixing matrix equation, (rotation transpose, bare mass matrix, propagating mass matrix)
    :rtype: tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix]]
    """
    rotation_transpose = rotation.transpose()
    bare_matrix = sym.Matrix(
        [
            [ALL_SYMBOL["bare"]["pi0"] ** 2, ALL_SYMBOL["bare"]["pi0-phi"] / 2, 0],
            [
                ALL_SYMBOL["bare"]["pi0-phi"] / 2,
                ALL_SYMBOL["bare"]["phi"] ** 2,
                ALL_SYMBOL["bare"]["eta-phi"] / 2,
            ],
            [0, ALL_SYMBOL["bare"]["eta-phi"] / 2, ALL_SYMBOL["bare"]["eta"] ** 2],
        ]
    )
    prop_matrix = sym.Matrix(
        [
            [ALL_SYMBOL["prop"]["pi0"] ** 2, 0, 0],
            [0, ALL_SYMBOL["prop"]["phi"] ** 2, 0],
            [0, 0, ALL_SYMBOL["prop"]["eta"] ** 2],
        ]
    )
    mixing_equation = sym.Eq(
        prop_matrix,
        sym.MatMul(rotation, bare_matrix, rotation_transpose),
        evaluate=False,
    )
    return mixing_equation, (rotation_transpose, bare_matrix, prop_matrix)
