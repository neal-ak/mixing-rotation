"""General use for symbolic calculations."""

# %%
import sympy as sym

PARTICLE_ORDER = "pi0", "eta", "etaP"


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
    for k in [
        r"\alpha",
        r"\beta",
        r"\gamma",
        r"\delta",
        r"\zeta",
        r"\kappa",
        "u",
        "v",
        "w",
    ]
}

# %%
MASS = {
    _preprocess_symbol(k): sym.Symbol(f"m_{{{k}}}", nonnegative=True)
    for k in [
        "u",
        "d",
        "s",
        r"\pi^{0}",
        r"\phi",
        r"\eta",
        r"\eta^{\prime}",
        r"A^{\prime}",
        r"\pi^{\pm}",
        r"K^{\pm}",
        r"\pi^{0}_{b}",
        r"\phi_{b}",
        r"\eta_{b}",
        r"\eta^{\prime}_{b}",
        r"\pi^{0} \phi",
        r"\eta \phi",
        r"\eta^{\prime} \phi",
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
        r"\phi_{b} \gamma",
        r"\pi^{0}_{b} \gamma",
        r"\eta_{b} \gamma",
        r"\eta^{\prime}_{b} \gamma",
        r"\phi_{b} \pi K",
        r"\pi^{0}_{b} \pi K",
        r"\eta_{b} \pi K",
        r"\eta^{\prime}_{b} \pi K",
        r"\phi \chi",
        r"\pi^{0} \chi",
        r"\eta \chi",
        r"\eta^{\prime} \chi",
        r"\phi \gamma",
        r"\pi^{0} \gamma",
        r"\eta \gamma",
        r"\eta^{\prime} \gamma",
        r"\phi \pi K",
        r"\pi^{0} \pi K",
        r"\eta \pi K",
        r"\eta^{\prime} \pi K",
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
        r"E_{K^\pm}",
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
def _create_nd_rotation(n_dim) -> tuple[sym.Matrix, list[sym.Matrix], sym.Matrix]:
    """Creates the n-D rotation matrix from 2D rotation matrices.

    :return: Rotation matrix, list of 2D rotations, generic 2D rotation
    :rtype: tuple[sym.Matrix, list[sym.Matrix], sym.Matrix]
    """
    theta = sym.Symbol(r"\theta", real=True)
    basic_2d = sym.Matrix(
        [
            [sym.cos(theta), -sym.sin(theta)],
            [sym.sin(theta), sym.cos(theta)],
        ]
    )
    index_pairs = []
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            index_pairs.append(tuple([i, j]))
    rotate = []
    for indices in index_pairs:
        matrix_2d = sym.Matrix.eye(n_dim)
        for i, a in enumerate(indices):
            for j, b in enumerate(indices):
                matrix_2d[a, b] = basic_2d[i, j]
        rotate.append(matrix_2d)
    for i, (m, s) in enumerate(
        zip(
            rotate,
            [ANGLE[k] for k in ["alpha", "beta", "gamma", "delta", "zeta", "kappa"]],
        )
    ):
        rotate[i] = m.subs(theta, s)
    result = 1
    for _x in rotate:
        result *= _x
    return result, rotate, basic_2d


# %%
def create_3d_rotation() -> (
    tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix], sym.Matrix]
):
    """Creates the 3D rotation matrix from three 2D rotation matrices.

    :return: Rotation matrix, (alpha, beta, gamma) rotations, 2D matrix
    :rtype: tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix], sym.Matrix]
    """
    rotation, matrices, basic_2d = _create_nd_rotation(3)
    return rotation, tuple(matrices), basic_2d


# %%
def create_4d_rotation() -> tuple[
    sym.Matrix,
    tuple[sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix],
    sym.Matrix,
]:
    """Creates the 4D rotation matrix from six 2D rotation matrices.

    :return: Rotation matrix, (alpha, beta, gamma, delta, zeta, kappa) rotations, 2D matrix
    :rtype: tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix], sym.Matrix]
    """
    rotation, matrices, basic_2d = _create_nd_rotation(4)
    return rotation, tuple(matrices), basic_2d
