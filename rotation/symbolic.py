"""Symbolically computes the mixing equation."""

# %%
import sympy as sym


# %%
def _preprocess_symbol(s):
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
            r"\phi \chi",
            r"\eta b \chi",
            r"\pi b \gamma",
            r"\phi \gamma",
            r"\eta b \gamma",
            r"\pi b \pi K",
            r"\phi \pi K",
            r"\eta b \pi K",
        ]
    },
    "rotated": {
        _preprocess_symbol(k): sym.Symbol(f"\\tilde{{g}}_{{{k}}}", real=True)
        for k in [
            r"\pi \chi",
            r"\phi^\prime \chi",
            r"\eta \chi",
            r"\pi \gamma",
            r"\phi^\prime \gamma",
            r"\eta \gamma",
            r"\pi \pi K",
            r"\phi^\prime \pi K",
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
    """
    Creates the mixing equation from the rotation, propagating, and bare matrices.
    Returns the equation, and a tuple of the matrices: (rotation_transpose,
    bare, propagating).
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


# %%
def create_kinetic():
    angles = [ALL_SYMBOL["angle"][k] for k in ["u", "v", "w"]]
    d = sym.sqrt(
        1
        - sym.sin(angles[0]) ** 2
        - sym.sin(angles[1]) ** 2
        - sym.sin(angles[2]) ** 2
        + 2 * sym.sin(angles[0]) * sym.sin(angles[1]) * sym.sin(angles[2])
    )
    return sym.Matrix(
        [
            [
                1,
                -sym.tan(angles[0]),
                (
                    sym.tan(angles[0]) * sym.sin(angles[2])
                    - sym.sin(angles[1]) / sym.cos(angles[0])
                )
                / d,
            ],
            [
                0,
                1 / sym.cos(angles[0]),
                (
                    sym.tan(angles[0]) * sym.sin(angles[1])
                    - sym.sin(angles[2]) / sym.cos(angles[0])
                )
                / d,
            ],
            [0, 0, sym.cos(angles[0]) / d],
        ]
    )


# %%
def create_axial_mix_equation(rotation, kinetic):
    rotation_transpose = rotation.transpose()
    kinetic_transpose = kinetic.transpose()
    bare_matrix = sym.Matrix(
        [
            [ALL_SYMBOL["bare"]["pi0"] ** 2, 0, 0],
            [0, ALL_SYMBOL["bare"]["eta"] ** 2, 0],
            [0, 0, ALL_SYMBOL["bare"]["eta0"] ** 2],
        ]
    )
    prop_matrix = sym.Matrix(
        [
            [ALL_SYMBOL["prop"]["pi0"] ** 2, 0, 0],
            [0, ALL_SYMBOL["prop"]["eta"] ** 2, 0],
            [0, 0, ALL_SYMBOL["prop"]["eta0"] ** 2],
        ]
    )
    mixing_equation = sym.Eq(
        prop_matrix,
        sym.MatMul(
            rotation, kinetic, bare_matrix, kinetic_transpose, rotation_transpose
        ),
        evaluate=False,
    )
    return mixing_equation, (
        kinetic_transpose,
        rotation_transpose,
        bare_matrix,
        prop_matrix,
    )


# %%
def _kinetic_sine(m1, m2):
    ma = ALL_SYMBOL["prop"]["A"]
    return (
        -m1 * m2 / (2 * ma**2 * sym.sqrt((1 - (m1 / ma) ** 2) * (1 - (m2 / ma) ** 2)))
    )


# %%
def kinetic_sines():
    s_u = _kinetic_sine(ALL_SYMBOL["bare"]["pi0-A"], ALL_SYMBOL["bare"]["eta-A"])
    s_v = _kinetic_sine(ALL_SYMBOL["bare"]["pi0-A"], ALL_SYMBOL["bare"]["eta0-A"])
    s_w = _kinetic_sine(ALL_SYMBOL["bare"]["eta0-A"], ALL_SYMBOL["bare"]["eta-A"])
    return s_u, s_v, s_w


# %%
def rotated_pseudo_meson_couplings() -> (
    tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
):
    """Symbolic expressions for the rotated meson to dark matter couplings.

    Using the pseudo scalar mediator phi.

    :return: pion-chi, mediator-chi, eta-chi
    :rtype: tuple[sympy.core.mul.Mul, sympy.core.mul.Mul, sympy.core.mul.Mul]
    """
    bare_couplings = sym.Matrix([[0, ALL_SYMBOL["coupling"]["phi-chi"], 0]])
    rotated_couplings = bare_couplings * create_3d_rotation()[0].transpose()
    return tuple(rotated_couplings)


# %%
def rotated_pseudo_photon_coupling() -> sym.core.mul.Mul:
    """Symbolic expression for the rotated mediator to photon coupling.

    Using the pseudo scalar mediator phi.

    Note that the bare un-rotated photon couplings have been eliminated from the
    final expression in favour of the known photon to pion and eta couplings.

    :return: mediator-photon
    :rtype: sym.core.mul.Mul
    """
    gamma_matrix = (
        sym.Matrix(
            [
                [
                    ALL_SYMBOL["coupling"]["pi-gamma"],
                    0,
                    ALL_SYMBOL["coupling"]["eta-gamma"],
                ]
            ]
        )
        * create_3d_rotation()[0].transpose()
    )
    gamma_eqns = [
        sym.Eq(g, e)
        for g, e in zip(
            [ALL_SYMBOL["rotated"][k] for k in ["pi-gamma", "phi-gamma", "eta-gamma"]],
            gamma_matrix,
        )
    ]
    gamma_solutions = sym.solve(
        gamma_eqns,
        [
            ALL_SYMBOL["rotated"]["phi-gamma"],
            ALL_SYMBOL["coupling"]["pi-gamma"],
            ALL_SYMBOL["coupling"]["eta-gamma"],
        ],
        dict=True,
    )
    return gamma_solutions[0][ALL_SYMBOL["rotated"]["phi-gamma"]].trigsimp()


# %%
def bare_pseudo_kaon_couplings() -> (
    tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
):
    """Symbolic expressions for the un-rotated kaon couplings.

    Using the pseudo scalar mediator phi.

    :return: pion-pion-kaon, eta-pion-kaon
    :rtype: tuple[sym.core.mul.Mul, sym.core.mul.Mul]
    """
    pi_pi_kaon = (
        ALL_SYMBOL["prop"]["K0"] ** 2
        - 2 * ALL_SYMBOL["prop"]["K0"] * ALL_SYMBOL["parameters"]["Epi"]
    )
    eta_pi_kaon = (
        ALL_SYMBOL["prop"]["K0"] ** 2
        - 2 * ALL_SYMBOL["prop"]["K0"] * ALL_SYMBOL["parameters"]["Epi"]
        - ALL_SYMBOL["prop"]["pi0"] ** 2
    ) / sym.sqrt(3)
    return pi_pi_kaon, eta_pi_kaon


# %%
def rotated_pseudo_kaon_couplings() -> (
    tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
):
    """Symbolic expressions for the rotated kaon couplings.

    Using the pseudo scalar mediator phi.

    :return: pion-pion-kaon, mediator-pion-kaon, eta-pion-kaon
    :rtype: tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
    """
    bare_couplings = sym.Matrix(
        [[ALL_SYMBOL["coupling"][f"{k}-pi-K"] for k in ["pi", "phi", "eta"]]]
    )
    rotated_couplings = bare_couplings * create_3d_rotation()[0].transpose()
    return tuple(rotated_couplings)
