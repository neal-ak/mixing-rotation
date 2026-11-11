"""Symbolic computations related to the axial vector mediator case."""

# %%
import sympy

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
def create_kinetic() -> sympy.Matrix:
    """Creates the kinetic mixing matrix for the axial vector mediator.

    For the axial vector mediator $A^\\prime$.
    Comes from `B8<https://arxiv.org/pdf/2611.xxxxx#equation.B.8>`__ in
    `arxiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :return: Kinetic mixing matrix.
    :rtype: sym.Matrix
    """
    angles = [ANGLE[k] for k in ["u", "v", "w"]]
    d = sympy.sqrt(
        1
        - sympy.sin(angles[0]) ** 2
        - sympy.sin(angles[1]) ** 2
        - sympy.sin(angles[2]) ** 2
        + 2 * sympy.sin(angles[0]) * sympy.sin(angles[1]) * sympy.sin(angles[2])
    )
    return sympy.Matrix(
        [
            [
                1,
                -sympy.tan(angles[0]),
                (
                    sympy.tan(angles[0]) * sympy.sin(angles[2])
                    - sympy.sin(angles[1]) / sympy.cos(angles[0])
                )
                / d,
            ],
            [
                0,
                1 / sympy.cos(angles[0]),
                (
                    sympy.tan(angles[0]) * sympy.sin(angles[1])
                    - sympy.sin(angles[2]) / sympy.cos(angles[0])
                )
                / d,
            ],
            [0, 0, sympy.cos(angles[0]) / d],
        ]
    )


# %%
def create_axial_mix_equation(
    rotation: sympy.Matrix, kinetic: sympy.Matrix
) -> tuple[sympy.Matrix, tuple[sympy.Matrix, sympy.Matrix, sympy.Matrix, sympy.Matrix]]:
    """Creates the axial vector mixing matrix equation.

    For the axial vector mediator $A^\\prime$.
    Comes from `B11<https://arxiv.org/pdf/2611.xxxxx#equation.B.11>`__ in
    `arxiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :param rotation: Rotation matrix.
    :type rotation: sym.Matrix
    :param kinetic: Kinetic rotation matrix.
    :type kinetic: sym.Matrix
    :return: Mixing matrix equation, tuple of kinetic transpose, rotation transpose, bare mass
    matrix, propagation mass matrix.
    :rtype: tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix]]
    """
    rotation_transpose = rotation.transpose()
    kinetic_transpose = kinetic.transpose()
    bare_matrix = sympy.Matrix(
        [
            [MASS["pi0Bare"] ** 2, 0, 0],
            [0, MASS["etaBare"] ** 2, 0],
            [0, 0, MASS["etaPBare"] ** 2],
        ]
    )
    prop_matrix = sympy.Matrix(
        [
            [MASS["pi0"] ** 2, 0, 0],
            [0, MASS["eta"] ** 2, 0],
            [0, 0, MASS["etaP"] ** 2],
        ]
    )
    mixing_equation = sympy.Eq(
        prop_matrix,
        sympy.MatMul(
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
def _kinetic_sine(m1, m2) -> sympy.core.mul.Mul:
    """Helper function in creating the sine of the kinetic mixing angle.

    $\frac{-m_1 m_2}{2 m_{A^{\\prime}}^2 \\sqrt{(1 - (m_1 / m_{A^{\\prime}}) ^ 2) (1 - (m_2 / m_{A^{\\prime}}) ^ 2)}}$

    :param m1: First off-diagonal mass.
    :type m1: type
    :param m2: Second off-diagonal mass.
    :type m2: type
    :return: Description of the return value.
    :rtype: sym.core.mul.Mul
    """
    ma = MASS["AP"]
    return (
        -m1 * m2 / (2 * ma**2 * sympy.sqrt((1 - (m1 / ma) ** 2) * (1 - (m2 / ma) ** 2)))
    )


# %%
def kinetic_sines() -> (
    tuple[sympy.core.mul.Mul, sympy.core.mul.Mul, sympy.core.mul.Mul]
):
    """Creates the sines of the three kinetic mixing angles.

    For the axial vector mediator $A^\\prime$.
    Comes from `B9<https://arxiv.org/pdf/2611.xxxxx#equation.B.9>`__ in
    `arxiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :return: Tuple of sine(u), sine(v), sine(w)
    :rtype: tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
    """
    s_u = _kinetic_sine(MASS["pi0-AP"], MASS["eta-AP"])
    s_v = _kinetic_sine(MASS["pi0-AP"], MASS["etaP-AP"])
    s_w = _kinetic_sine(MASS["etaP-AP"], MASS["eta-AP"])
    return s_u, s_v, s_w


# %%
def bare_axial_meson_couplings() -> dict[sympy.Symbol, sympy.core.mul.Mul]:
    """Symbolic expressions for the un-rotated meson to dark matter couplings.

    For the axial vector mediator $A^\\prime$.
    Implicit in `B13<https://arxiv.org/pdf/2611.xxxxx#equation.B.13>`__ in
    `arxiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :return: Bare meson dark matter couplings, keys are the coupling symbols.
    :rtype: dict[sym.Symbol, sym.core.mul.Mul]
    """
    couplings = {
        AXIAL_COUPLING[m + "-chi"]: AXIAL_COUPLING["a"]
        * MASS[f"{m}-AP"]
        / MASS["AP"] ** 2
        / sympy.sqrt(1 - (MASS[f"{m}-AP"] / MASS["AP"]) ** 2)
        for m in PARTICLE_ORDER
    }
    return couplings


# %%
def rotated_axial_meson_couplings() -> dict[sympy.Symbol, sympy.core.mul.Mul]:
    """Symbolic expressions for the rotated meson to dark matter couplings.

    For the axial vector mediator $A^\\prime$.
    Comes from `B13<https://arxiv.org/pdf/2611.xxxxx#equation.B.13>`__ in
    `arxiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :return: Rotated meson dark matter couplings, keys are the coupling symbols.
    :rtype: dict[sym.Symbol, sym.core.mul.Mul]
    """
    bares = bare_axial_meson_couplings()
    bare_couplings = sympy.Matrix(
        [[*[bares[AXIAL_COUPLING[x + "-chi"]] for x in PARTICLE_ORDER]]]
    )
    rotated_couplings = (
        bare_couplings * create_kinetic() * create_3d_rotation()[0].transpose()
    )
    return dict(
        [*zip([AXIAL_COUPLING[x + "-chi"] for x in PARTICLE_ORDER], rotated_couplings)]
    )


# %%
def bare_axial_masses() -> dict[sympy.Symbol, sympy.core.mul.Mul]:
    """The bare axial mediator off-diagonal mass terms.

    For the axial vector mediator $A^\\prime$.
    Comes from `B5<https://arxiv.org/pdf/2611.xxxxx#equation.B.5>`__ in
    `arxiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :return: Bare particle off-diagonal masses, keys are the mass symbols.
    :rtype: dict[sym.Symbol, sym.core.mul.Mul]
    """
    terms = {
        MASS["pi0-AP"]: (
            PARAMETERS["au"] - PARAMETERS["ad"],
            PARAMETERS["au"] * MASS["u"] - PARAMETERS["ad"] * MASS["d"],
        ),
        MASS["eta-AP"]: (
            1
            / sympy.sqrt(3)
            * (PARAMETERS["au"] + PARAMETERS["ad"] - 2 * PARAMETERS["as"]),
            1
            / sympy.sqrt(3)
            * (
                PARAMETERS["au"] * MASS["u"]
                + PARAMETERS["ad"] * MASS["d"]
                - 2 * PARAMETERS["as"] * MASS["s"]
            ),
        ),
        MASS["etaP-AP"]: (
            sympy.sqrt(6)
            / 3
            * (PARAMETERS["au"] + PARAMETERS["ad"] + PARAMETERS["as"]),
            sympy.sqrt(6)
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
def bare_axial_kaon_couplings() -> dict[sympy.Symbol, sympy.core.mul.Mul]:
    """Symbolic expressions for the un-rotated kaon couplings.

    For the axial vector mediator $A^\\prime$.
    Implicit in `D24<https://arxiv.org/pdf/2611.xxxxx#equation.D.24>`__ in
    `arxiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :return: Bare kaon couplings, keys are the coupling symbols.
    :rtype: dict[sym.Symbol, sym.core.mul.Mul]
    """
    # helpers
    lambda_axial_pion_kaon_j_over_m_j = -(
        (2 * sympy.I / MASS["AP"] ** 2)
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
            sympy.I
            * PARAMETERS["F"] ** -3
            * PARAMETERS["g8"]
            * (-MASS["Kpm"] ** 2 + 2 * MASS["Kpm"] * PARAMETERS["Epipm"])
            + lambda_axial_pion_kaon_j_over_m_j * MASS["pi0-AP"]
        ),
        "eta": (
            sympy.sqrt(3) ** -1
            * sympy.I
            * PARAMETERS["F"] ** -3
            * (2 * PARAMETERS["g8"])
            * (
                MASS["Kpm"] ** 2
                - 6 * MASS["Kpm"] * PARAMETERS["Epipm"]
                + 2 * MASS["pipm"] ** 2
            )
            + lambda_axial_pion_kaon_j_over_m_j * MASS["eta-AP"]
        ),
        "etaP": (
            sympy.sqrt(2)
            * sympy.sqrt(3) ** -1
            * sympy.I
            * PARAMETERS["F"] ** -3
            * (2 * PARAMETERS["g8"] + 3 * PARAMETERS["g8s"])
            * (-MASS["Kpm"] ** 2 + MASS["pipm"] ** 2)
            + lambda_axial_pion_kaon_j_over_m_j * MASS["etaP-AP"]
        ),
    }

    bare = {
        AXIAL_COUPLING[k + "Bare-pi-K"]: l
        / sympy.sqrt(1 - (MASS[f"{k}-AP"] / MASS["AP"]) ** 2)
        for k, l in couplings.items()
    }
    return bare


# %%
def rotated_axial_kaon_couplings() -> dict[sympy.Symbol, sympy.core.mul.Mul]:
    """Symbolic expressions for the rotated kaon couplings.

    For the axial vector mediator $A^\\prime$.
    Comes from `D24<https://arxiv.org/pdf/2611.xxxxx#equation.D.24>`__ in
    `arxiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :return: Rotated particle couplings, keys are the coupling symbols.
    :rtype: dict[sympy.Symbol, sympy.core.mul.Mul]
    """
    bare_couplings = sympy.Matrix(
        [[AXIAL_COUPLING[f"{k}Bare-pi-K"] for k in PARTICLE_ORDER]]
    )
    rotated_couplings = (
        bare_couplings * create_kinetic() * create_3d_rotation()[0].transpose()
    )
    return dict(
        [*zip([AXIAL_COUPLING[x + "-pi-K"] for x in PARTICLE_ORDER], rotated_couplings)]
    )
