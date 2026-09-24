"""Symbolically computes the axial mixing equation."""

# %%
import sympy as sym

# %%
from .general import (
    PARTICLE_ORDER,
    ANGLE,
    AXIAL_COUPLING,
    MASS,
    PARAMETERS,
    create_3d_rotation,
)


# %%
def create_kinetic() -> sym.Matrix:
    """Creates the kinetic mixing matrix for the axial vector mediator.

    :return: Kinetic mixing matrix
    :rtype: sym.Matrix
    """
    angles = [ANGLE[k] for k in ["u", "v", "w"]]
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
def create_axial_mix_equation(
    rotation: sym.Matrix, kinetic: sym.Matrix
) -> tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix]]:
    """Creates the axial vector mixing matrix equation.

    :param rotation: Rotation matrix
    :type rotation: sym.Matrix
    :param kinetic: Kinetic rotation matrix
    :type kinetic: sym.Matrix
    :return: Mixing matrix equation, (kinetic transpose, rotation transpose, bare mass matrix, propagation mass matrix)
    :rtype: tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix]]
    """
    rotation_transpose = rotation.transpose()
    kinetic_transpose = kinetic.transpose()
    bare_matrix = sym.Matrix(
        [
            [MASS["pi0Bare"] ** 2, 0, 0],
            [0, MASS["etaBare"] ** 2, 0],
            [0, 0, MASS["etaPBare"] ** 2],
        ]
    )
    prop_matrix = sym.Matrix(
        [
            [MASS["pi0"] ** 2, 0, 0],
            [0, MASS["eta"] ** 2, 0],
            [0, 0, MASS["etaP"] ** 2],
        ]
    )
    mixing_equation = sym.Eq(
        prop_matrix,
        sym.MatMul(
            rotation, kinetic_transpose, bare_matrix, kinetic, rotation_transpose
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
def _kinetic_sine(m1, m2) -> sym.core.mul.Mul:
    """Helper to automate sine of kinetic mixing angle."""
    ma = MASS["AP"]
    return (
        -m1 * m2 / (2 * ma**2 * sym.sqrt((1 - (m1 / ma) ** 2) * (1 - (m2 / ma) ** 2)))
    )


# %%
def kinetic_sines() -> tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]:
    """Creates the sines of the three kinetic mixing angles.

    :return: sine(u), sine(v), sine(w)
    :rtype: tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
    """
    s_u = _kinetic_sine(MASS["pi0-AP"], MASS["eta-AP"])
    s_v = _kinetic_sine(MASS["pi0-AP"], MASS["etaP-AP"])
    s_w = _kinetic_sine(MASS["etaP-AP"], MASS["eta-AP"])
    return s_u, s_v, s_w


# %%
def bare_axial_meson_couplings() -> dict[sym.Symbol, sym.core.mul.Mul]:
    """Symbolic expressions for the un-rotated meson to dark matter couplings.

    Using the axial vector mediator A.

    :return: bare particle couplings
    :rtype: dict[sympy.Symbol, sympy.core.mul.Mul]
    """
    couplings = {
        AXIAL_COUPLING[m + "Bare-pi-K"]: AXIAL_COUPLING["a"]
        * MASS[f"{m}-AP"]
        / MASS["AP"] ** 2
        / sym.sqrt(1 - (MASS[f"{m}-AP"] / MASS["AP"]) ** 2)
        for m in PARTICLE_ORDER
    }
    return couplings


# %%
def rotated_axial_meson_couplings() -> dict[sym.Symbol, sym.core.mul.Mul]:
    """Symbolic expressions for the rotated meson to dark matter couplings.

    Using the axial vector mediator A.

    :return: rotated particle couplings
    :rtype: dict[sympy.Symbol, sympy.core.mul.Mul]
    """
    bares = bare_axial_meson_couplings()
    bare_couplings = sym.Matrix(
        [[*[bares[AXIAL_COUPLING[x + "Bare-pi-K"]] for x in PARTICLE_ORDER]]]
    )
    rotated_couplings = (
        bare_couplings * create_kinetic() * create_3d_rotation()[0].transpose()
    )
    return dict(
        [*zip([AXIAL_COUPLING[x + "-chi"] for x in PARTICLE_ORDER], rotated_couplings)]
    )


# %%
def bare_axial_masses() -> dict[sym.Symbol, sym.core.mul.Mul]:
    """The bare axial mediator off-diagonal terms.

    :return: bare particle masses
    :rtype: tuple[sympy.Symbol, sympy.core.mul.Mul]
    """
    terms = {
        MASS["pi0-AP"]: (
            PARAMETERS["au"] - PARAMETERS["ad"],
            PARAMETERS["au"] * MASS["u"] - PARAMETERS["ad"] * MASS["d"],
        ),
        MASS["eta-AP"]: (
            1
            / sym.sqrt(3)
            * (PARAMETERS["au"] + PARAMETERS["ad"] - 2 * PARAMETERS["as"]),
            1
            / sym.sqrt(3)
            * (
                PARAMETERS["au"] * MASS["u"]
                + PARAMETERS["ad"] * MASS["d"]
                - 2 * PARAMETERS["as"] * MASS["s"]
            ),
        ),
        MASS["etaP-AP"]: (
            sym.sqrt(6) / 3 * (PARAMETERS["au"] + PARAMETERS["ad"] + PARAMETERS["as"]),
            sym.sqrt(6)
            / 3
            * (
                PARAMETERS["au"] * MASS["u"]
                + PARAMETERS["ad"] * MASS["d"]
                + PARAMETERS["as"] * MASS["s"]
            ),
        ),
    }
    bare = {
        k: PARAMETERS["F"]
        * (
            a
            * (
                1
                + 16
                * PARAMETERS["B0"]
                * PARAMETERS["L4"]
                / PARAMETERS["F"] ** 2
                * (MASS["u"] + MASS["d"] + MASS["s"])
            )
            + b * 16 * PARAMETERS["B0"] * PARAMETERS["L5"] / PARAMETERS["F"] ** 2
        )
        for k, (a, b) in terms.items()
    }
    return bare


# %%
def bare_axial_kaon_couplings() -> dict[sym.Symbol, sym.core.mul.Mul]:
    """Symbolic expressions for the un-rotated kaon couplings.

    Using the axial vector mediator A.

    :return: bare particle couplings
    :rtype: dict[sympy.Symbol, sympy.core.mul.Mul]
    """
    # helpers
    L_AprimepiKj_over_m_j = -(
        (2 * sym.I / MASS["AP"] ** 2)
        / PARAMETERS["F"] ** 2
        * (
            PARAMETERS["g8"]
            * (
                (PARAMETERS["ad"] + PARAMETERS["au"])
                * (PARAMETERS["Epipm"] * MASS["Kpm"] - MASS["pipm"] ** 2)
                - (PARAMETERS["as"] + PARAMETERS["au"])
                * (PARAMETERS["Epipm"] * MASS["Kpm"] - MASS["Kpm"] ** 2)
            )
            + PARAMETERS["g8s"]
            * (PARAMETERS["au"] + PARAMETERS["ad"] + PARAMETERS["as"])
            * (MASS["Kpm"] ** 2 - MASS["pipm"] ** 2)
        )
    )

    couplings = {
        "pi0": (
            sym.I
            * PARAMETERS["F"] ** -3
            * PARAMETERS["g8"]
            * (-MASS["Kpm"] ** 2 + 2 * MASS["Kpm"] * PARAMETERS["Epipm"])
            + L_AprimepiKj_over_m_j * MASS["pi0-AP"]
        ),
        "eta": (
            sym.sqrt(3) ** -1
            * sym.I
            * PARAMETERS["F"] ** -3
            * (2 * PARAMETERS["g8"])
            * (
                MASS["Kpm"] ** 2
                - 6 * MASS["Kpm"] * PARAMETERS["Epipm"]
                + 2 * MASS["pipm"] ** 2
            )
            + L_AprimepiKj_over_m_j * MASS["eta-AP"]
        ),
        "etaP": (
            sym.sqrt(2)
            * sym.sqrt(3) ** -1
            * sym.I
            * PARAMETERS["F"] ** -3
            * (2 * PARAMETERS["g8"] + 3 * PARAMETERS["g8s"])
            * (-MASS["Kpm"] ** 2 + MASS["pipm"] ** 2)
            + L_AprimepiKj_over_m_j * MASS["etaP-AP"]
        ),
    }

    bare = {
        AXIAL_COUPLING[k + "Bare-pi-K"]: l
        / sym.sqrt(1 - (MASS[f"{k}-AP"] / MASS["AP"]) ** 2)
        for k, l in couplings.items()
    }
    return bare


# %%
def rotated_axial_kaon_couplings() -> dict[sym.Symbol, sym.core.mul.Mul]:
    """Symbolic expressions for the rotated kaon couplings.

    Using the axial vector mediator A.

    :return: rotated particle couplings
    :rtype: dict[sympy.Symbol, sympy.core.mul.Mul]
    """
    bare_couplings = sym.Matrix(
        [[AXIAL_COUPLING[f"{k}Bare-pi-K"] for k in PARTICLE_ORDER]]
    )
    rotated_couplings = (
        bare_couplings * create_kinetic() * create_3d_rotation()[0].transpose()
    )
    return dict(
        [*zip([AXIAL_COUPLING[x + "-pi-K"] for x in PARTICLE_ORDER], rotated_couplings)]
    )
