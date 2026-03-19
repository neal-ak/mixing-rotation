"""Symbolically computes the axial mixing equation."""

# %%
import sympy as sym

# %%
from .general import (
    ANGLE,
    AXIAL_COUPLING,
    BARE,
    PARAMETERS,
    PROPAGATION,
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
            [BARE["pi0Bare"] ** 2, 0, 0],
            [0, BARE["etaBare"] ** 2, 0],
            [0, 0, BARE["etaPBare"] ** 2],
        ]
    )
    prop_matrix = sym.Matrix(
        [
            [PROPAGATION["pi0"] ** 2, 0, 0],
            [0, PROPAGATION["eta"] ** 2, 0],
            [0, 0, PROPAGATION["etaP"] ** 2],
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
    ma = PROPAGATION["AP"]
    return (
        -m1 * m2 / (2 * ma**2 * sym.sqrt((1 - (m1 / ma) ** 2) * (1 - (m2 / ma) ** 2)))
    )


# %%
def kinetic_sines() -> tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]:
    """Creates the sines of the three kinetic mixing angles.

    :return: sine(u), sine(v), sine(w)
    :rtype: tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
    """
    s_u = _kinetic_sine(BARE["pi0-AP"], BARE["eta-AP"])
    s_v = _kinetic_sine(BARE["pi0-AP"], BARE["etaP-AP"])
    s_w = _kinetic_sine(BARE["etaP-AP"], BARE["eta-AP"])
    return s_u, s_v, s_w


# %%
def bare_axial_meson_couplings() -> (
    tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
):
    """Symbolic expressions for the un-rotated meson to dark matter couplings.

    Using the axial vector mediator A.

    :return: pion-chi, eta-chi, eta^prime-chi
    :rtype: tuple[sympy.core.mul.Mul, sympy.core.mul.Mul, sympy.core.mul.Mul]
    """
    masses = [BARE[f"{k}-AP"] for k in ["pi0", "eta", "etaP"]]
    return tuple(
        [
            AXIAL_COUPLING["a"]
            * m
            / PROPAGATION["AP"] ** 2
            / sym.sqrt(1 - (m / PROPAGATION["AP"]) ** 2)
            for m in masses
        ]
    )


# %%
def rotated_axial_meson_couplings() -> (
    tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
):
    """Symbolic expressions for the rotated meson to dark matter couplings.

    Using the axial vector mediator A.

    :return: pion-chi, eta-chi, eta^prime-chi
    :rtype: tuple[sympy.core.mul.Mul, sympy.core.mul.Mul, sympy.core.mul.Mul]
    """
    bare_couplings = sym.Matrix([[*bare_axial_meson_couplings()]])
    rotated_couplings = (
        bare_couplings * create_kinetic() * create_3d_rotation()[0].transpose()
    )
    return tuple(rotated_couplings)


# %%
def bare_axial_masses() -> tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]:
    """The bare axial mediator off-diagonal terms.

    :return: pion-mediator, eta-mediator, eta^prime-mediator
    :rtype: tuple[sympy.core.mul.Mul, sympy.core.mul.Mul, sympy.core.mul.Mul]
    """
    terms = [
        (
            PARAMETERS["au"] - PARAMETERS["ad"],
            PARAMETERS["au"] * PARAMETERS["mu"] - PARAMETERS["ad"] * PARAMETERS["md"],
        ),
        (
            1
            / sym.sqrt(3)
            * (PARAMETERS["au"] + PARAMETERS["ad"] - 2 * PARAMETERS["as"]),
            1
            / sym.sqrt(3)
            * (
                PARAMETERS["au"] * PARAMETERS["mu"]
                + PARAMETERS["ad"] * PARAMETERS["md"]
                - 2 * PARAMETERS["as"] * PARAMETERS["ms"]
            ),
        ),
        (
            sym.sqrt(6) / 3 * (PARAMETERS["au"] + PARAMETERS["ad"] + PARAMETERS["as"]),
            sym.sqrt(6)
            / 3
            * (
                PARAMETERS["au"] * PARAMETERS["mu"]
                + PARAMETERS["ad"] * PARAMETERS["md"]
                + PARAMETERS["as"] * PARAMETERS["ms"]
            ),
        ),
    ]
    return tuple(
        [
            PARAMETERS["F"]
            * (
                a
                * (
                    1
                    + 16
                    * PARAMETERS["B0"]
                    * PARAMETERS["L4"]
                    / PARAMETERS["F"] ** 2
                    * (PARAMETERS["mu"] + PARAMETERS["md"] + PARAMETERS["ms"])
                )
                + b * 16 * PARAMETERS["B0"] * PARAMETERS["L5"] / PARAMETERS["F"] ** 2
            )
            for a, b in terms
        ]
    )


# %%
def bare_axial_kaon_couplings() -> (
    tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
):
    """Symbolic expressions for the un-rotated kaon couplings.

    Using the axial vector mediator A.

    :return: pion-pion-kaon, eta-pion-kaon, eta^prime-pion-kaon
    :rtype: tuple[sympy.core.mul.Mul, sympy.core.mul.Mul, sympy.core.mul.Mul]
    """
    pi_pi_k = (
        -(
            (
                sym.I
                * PROPAGATION["Kpm"]
                * (
                    PARAMETERS["g8"] * PROPAGATION["AP"] ** 2
                    - 2
                    * PARAMETERS["F"]
                    * (
                        (PARAMETERS["ad"] + PARAMETERS["au"]) * PARAMETERS["g8"]
                        + (PARAMETERS["ad"] + PARAMETERS["as"] + PARAMETERS["au"])
                        * PARAMETERS["g8s"]
                    )
                    * BARE["pi0-AP"] ** 2
                )
                * PARAMETERS["Epipm"]
            )
            / (PARAMETERS["F"] ** 3 * PROPAGATION["AP"] ** 2)
        )
        + (
            sym.I
            * PARAMETERS["g8"]
            * PROPAGATION["Kpm"]
            * (
                PROPAGATION["AP"] ** 2
                + 2
                * (PARAMETERS["ad"] - PARAMETERS["as"])
                * PARAMETERS["F"]
                * BARE["pi0-AP"] ** 2
            )
            * PARAMETERS["Echi"]
        )
        / (PARAMETERS["F"] ** 3 * PROPAGATION["AP"] ** 2)
        - (
            sym.I
            * (
                (PARAMETERS["ad"] + PARAMETERS["au"]) * PARAMETERS["g8"]
                + (PARAMETERS["ad"] + PARAMETERS["as"] + PARAMETERS["au"])
                * PARAMETERS["g8s"]
            )
            * (PROPAGATION["Kpm"] ** 2 + PROPAGATION["pipm"] ** 2)
            * BARE["pi0-AP"] ** 2
        )
        / (PARAMETERS["F"] ** 2 * PROPAGATION["AP"] ** 2)
    )
    eta_pi_k = (
        (
            sym.I
            * PROPAGATION["Kpm"]
            * (
                -sym.sqrt(3) * PARAMETERS["g8"] * PROPAGATION["AP"] ** 2
                + 6
                * (PARAMETERS["ad"] + PARAMETERS["au"])
                * PARAMETERS["F"]
                * PARAMETERS["g8"]
                * BARE["eta-AP"] ** 2
                + 6
                * (PARAMETERS["ad"] + PARAMETERS["as"] + PARAMETERS["au"])
                * PARAMETERS["F"]
                * PARAMETERS["g8s"]
                * BARE["eta-AP"] ** 2
            )
            * PARAMETERS["Epipm"]
        )
        / (3 * PARAMETERS["F"] ** 3 * PROPAGATION["AP"] ** 2)
        - (
            sym.I
            * PARAMETERS["g8"]
            * PROPAGATION["Kpm"]
            * (
                sym.sqrt(3) * PROPAGATION["AP"] ** 2
                + 2
                * (-PARAMETERS["ad"] + PARAMETERS["as"])
                * PARAMETERS["F"]
                * BARE["eta-AP"] ** 2
            )
            * PARAMETERS["Echi"]
        )
        / (PARAMETERS["F"] ** 3 * PROPAGATION["AP"] ** 2)
        + (
            sym.I
            * (
                sym.sqrt(3) * PARAMETERS["g8"] * PROPAGATION["AP"] ** 2
                - 3
                * (PARAMETERS["ad"] + PARAMETERS["au"])
                * PARAMETERS["F"]
                * PARAMETERS["g8"]
                * BARE["eta-AP"] ** 2
                - 3
                * (PARAMETERS["ad"] + PARAMETERS["as"] + PARAMETERS["au"])
                * PARAMETERS["F"]
                * PARAMETERS["g8s"]
                * BARE["eta-AP"] ** 2
            )
            * (PROPAGATION["Kpm"] ** 2 + PROPAGATION["pipm"] ** 2)
        )
        / (3 * PARAMETERS["F"] ** 3 * PROPAGATION["AP"] ** 2)
    )
    etaP_pi_k = (
        1
        / (6 * PARAMETERS["F"] ** 3 * PROPAGATION["AP"] ** 2)
        * sym.I
        * (
            2 * sym.sqrt(6) * PARAMETERS["g8"] * PROPAGATION["AP"] ** 2
            + 3 * sym.sqrt(6) * PARAMETERS["g8s"] * PROPAGATION["AP"] ** 2
            - 6
            * (PARAMETERS["ad"] + PARAMETERS["au"])
            * PARAMETERS["F"]
            * PARAMETERS["g8"]
            * BARE["etaP-AP"] ** 2
            - 6
            * (PARAMETERS["ad"] + PARAMETERS["as"] + PARAMETERS["au"])
            * PARAMETERS["F"]
            * PARAMETERS["g8s"]
            * BARE["etaP-AP"] ** 2
        )
        * (PROPAGATION["Kpm"] ** 2 + PROPAGATION["pipm"] ** 2)
        + (
            2
            * sym.I
            * (PARAMETERS["ad"] - PARAMETERS["as"])
            * PARAMETERS["g8"]
            * PROPAGATION["Kpm"]
            * BARE["etaP-AP"] ** 2
            * PARAMETERS["Echi"]
        )
        / (PARAMETERS["F"] ** 2 * PROPAGATION["AP"] ** 2)
        + (
            sym.I
            * PROPAGATION["Kpm"]
            * (
                -2 * sym.sqrt(6) * PARAMETERS["g8"] * PROPAGATION["AP"] ** 2
                - 3 * sym.sqrt(6) * PARAMETERS["g8s"] * PROPAGATION["AP"] ** 2
                + 6
                * (PARAMETERS["ad"] + PARAMETERS["au"])
                * PARAMETERS["F"]
                * PARAMETERS["g8"]
                * BARE["etaP-AP"] ** 2
                + 6
                * (PARAMETERS["ad"] + PARAMETERS["as"] + PARAMETERS["au"])
                * PARAMETERS["F"]
                * PARAMETERS["g8s"]
                * BARE["etaP-AP"] ** 2
            )
            * PARAMETERS["Epipm"]
        )
        / (3 * PARAMETERS["F"] ** 3 * PROPAGATION["AP"] ** 2)
    )
    return tuple(
        [
            l / sym.sqrt(1 - (m / PROPAGATION["AP"]) ** 2)
            for m, l in zip(
                [BARE[f"{k}-AP"] for k in ["pi0", "eta", "etaP"]],
                [pi_pi_k, eta_pi_k, etaP_pi_k],
            )
        ]
    )


# %%
def rotated_axial_kaon_couplings() -> (
    tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
):
    """Symbolic expressions for the rotated kaon couplings.

    Using the axial vector mediator A.

    :return: pion-pion-kaon, eta-pion-kaon, eta^prime-pion-kaon
    :rtype: tuple[sympy.core.mul.Mul, sympy.core.mul.Mul, sympy.core.mul.Mul]
    """
    bare_couplings = sym.Matrix(
        [[AXIAL_COUPLING[f"{k}Bare-pi-K"] for k in ["pi0", "eta", "etaP"]]]
    )
    rotated_couplings = (
        bare_couplings * create_kinetic() * create_3d_rotation()[0].transpose()
    )
    return tuple(rotated_couplings)
