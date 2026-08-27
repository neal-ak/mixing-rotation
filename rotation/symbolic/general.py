"""General use for symbolic calculations."""

# %%
import sympy as sym


# %%
def _preprocess_symbol(s):
    """Helper to automate assigning dict key strings from symbols."""
    s = str(s).replace("\\", "")
    s = s.replace(r"_{b}", "Bare")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace("^", "")
    s = s.replace("_", "")
    s = s.replace("tilde", "")
    s = s.replace("prime", "P")
    return s.strip().replace(" ", "-")


# %%
ANGLE = {
    _preprocess_symbol(k): sym.Symbol(k, real=True)
    for k in [r"\alpha", r"\beta", r"\gamma", "u", "v", "w"]
}

# %%
MASS = {
    _preprocess_symbol(k): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
    for k in [
        "u",
        "d",
        "s",
        r"\pi^{0}",
        r"\tilde{\phi}",
        r"\eta",
        r"\eta^{\prime}",
        r"A^{\prime}",
        r"\pi^{\pm}",
        r"K^{\pm}",
        r"\pi^{0}_{b}",
        r"\tilde{\phi}_{b}",
        r"\eta_{b}",
        r"\pi^{0} \tilde{\phi}",
        r"\eta \tilde{\phi}",
        r"\eta^{\prime}_{b}",
        r"\pi^{0} A^{\prime}",
        r"\eta A^{\prime}",
        r"\eta^{\prime} A^{\prime}",
    ]
}

# %%
COUPLING = {
    _preprocess_symbol(k): sym.Symbol(f"\\tilde{{g}}_{{{k}}}", real=True)
    for k in [
        "u",
        "d",
        "s",
        r"\chi",
        r"\pi^{0}_{b} \chi",
        r"\eta_{b} \chi",
        r"\pi^{0}_{b} \gamma",
        r"\tilde{\phi}_{b} \gamma",
        r"\eta_{b} \gamma",
        r"\pi^{0}_{b} \pi K",
        r"\tilde{\phi}_{b} \pi K",
        r"\eta_{b} \pi K",
        r"\pi^{0} \chi",
        r"\tilde{\phi} \chi",
        r"\eta \chi",
        r"\pi^{0} \gamma",
        r"\tilde{\phi} \gamma",
        r"\eta \gamma",
        r"\pi^{0} \pi K",
        r"\tilde{\phi} \pi K",
        r"\eta \pi K",
    ]
}

# %%
AXIAL_COUPLING = {
    _preprocess_symbol(k): sym.Symbol(f"\\lambda_{{{k}}}", real=True)
    for k in [
        "a",
        r"\pi^{0}_{b} \chi",
        r"\eta^{\prime}_{b} \chi",
        r"\eta_{b} \chi",
        r"\pi^{0}_{b} \pi K",
        r"\eta_{b} \pi K",
        r"\eta^{\prime}_{b} \pi K",
        r"\pi^{0} \chi",
        r"\eta^{\prime} \chi",
        r"\eta \chi",
        r"\pi^{0} \pi K",
        r"\eta \pi K",
        r"\eta^{\prime} \pi K",
    ]
}

# %%
PARAMETERS = {
    _preprocess_symbol(k): sym.Symbol(k, real=True)
    for k in [
        r"E_{\pi^\pm}",
        r"E_{\chi}",
        "F",
        r"B_{0}",
        r"L_{4}",
        r"L_{5}",
        r"L_{6}",
        r"L_{7}",
        r"a_{u}",
        r"a_{d}",
        r"a_{s}",
        r"g_{8}",
        r"g_{8}^{s}",
    ]
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
        zip(rotate, [ANGLE[k] for k in ["alpha", "beta", "gamma"]])
    ):
        rotate[i] = m.subs(theta, s)
    return rotate[0] * rotate[1] * rotate[2], tuple(rotate), basic_2d
