"""General use for symbolic calculations."""

# %%
import sympy as sym


# %%
def _preprocess_symbol(s):
    """Helper to automate assigning dict key strings from symbols."""
    s = str(s).replace("\\", "")
    s = (
        s.replace("eta^prime", "etap")
        .replace("^prime", "")
        .replace("^", "")
        .replace("_", "")
    )
    s = s.replace(" b", "")
    s = s.replace("tilde", "").replace("{", "").replace("}", "")
    return s.strip().replace(" ", "-")


# %%
ANGLE = {
    _preprocess_symbol(k): sym.Symbol(k, real=True)
    for k in [r"\alpha", r"\beta", r"\gamma", "u", "v", "w"]
}

# %%
PROPAGATION = {
    _preprocess_symbol(k): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
    for k in [
        r"\pi^0",
        r"\tilde{\phi}^\prime",
        r"\eta",
        r"\eta_0",
        r"A^\prime",
        r"K^0",
        r"\pi",
        "K",
    ]
}

# %%
BARE = {
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
        r"A",
        r"\eta^\prime A",
    ]
}

# %%
COUPLING = {
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
}

# %%
ROTATED = {
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
}

# %%
AXIAL_COUPLING = {
    _preprocess_symbol(k): sym.Symbol(f"\\lambda_{{{k}}}", real=True)
    for k in [
        "a",
        r"\pi b \chi",
        r"\eta^\prime b \chi",
        r"\eta b \chi",
        r"\pi \pi K b",
        r"\eta \pi K b",
        r"\eta^\prime \pi K b",
    ]
}

# %%
AXIAL_ROTATED = {
    _preprocess_symbol(k): sym.Symbol(f"\\lambda_{{{k}}}", real=True)
    for k in [
        r"\pi \chi",
        r"\eta^\prime \chi",
        r"\eta \chi",
        r"\pi \pi K",
        r"\eta \pi K",
        r"\eta^\prime \pi K",
    ]
}

# %%
PARAMETERS = {
    _preprocess_symbol(k): sym.Symbol(k, real=True)
    for k in [
        r"E_\pi",
        r"F_0",
        r"B_0",
        r"L_4",
        r"L_5",
        r"a_u",
        r"a_d",
        r"a_s",
        r"m_u",
        r"m_d",
        r"m_s",
        r"g_8",
        r"g_{8s}",
        r"E_\chi",
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
