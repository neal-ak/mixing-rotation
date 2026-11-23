"""Symbolic computations related to the pseudoscalar mediator case."""

# %%
import sympy

# %%
from .general import (
    PARTICLE_ORDER,
    MASS,
    COUPLING,
    PARAMETERS,
    _create_nd_rotation,
)


# %%
def create_mix_equation(
    rotation: sympy.Matrix,
) -> tuple[sympy.Matrix, tuple[sympy.Matrix, sympy.Matrix, sympy.Matrix]]:
    """Creates the symbolic mixing equation.

    For the pseudoscalar mediator $\tilde{\\phi}$.
    Comes from `A9<https://arxiv.org/pdf/2611.xxxxx#equation.A.9>`__ in
    `arXiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :param rotation: The full rotation matrix.
    :type rotation: sym.Matrix
    :return: Mixing equation, and tuple of the rotation matrix transpose, the bare mass matrix, and
    the propagating mass matrix.
    :rtype: tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix]]
    :raises ValueError: Raised when the rotation matrix is not square.
    """
    if not rotation.rows == rotation.cols:
        raise ValueError("Rotation matrix must be square")
    particles = ["phi"] + list(PARTICLE_ORDER[: rotation.rows - 1])
    phi_index = particles.index("phi")
    rotation_transpose = rotation.transpose()
    bare_matrix = sympy.Matrix.diag([MASS[k + "Bare"] ** 2 for k in particles])
    bare_matrix[:, phi_index] = sympy.Matrix(
        [
            MASS[f"{k}-phi"] ** 2 / 2 if k != "phi" else MASS[f"{k}Bare"] ** 2
            for k in particles
        ]
    )
    bare_matrix[phi_index, :] = bare_matrix[:, phi_index].T
    prop_matrix = sympy.Matrix.diag([MASS[k] ** 2 for k in particles])
    mixing_equation = sympy.Eq(
        prop_matrix,
        sympy.MatMul(rotation, bare_matrix, rotation_transpose),
        evaluate=False,
    )
    return mixing_equation, (rotation_transpose, bare_matrix, prop_matrix)


# %%
def rotated_pseudo_meson_couplings(
    n_dim: int,
) -> dict[sympy.Symbol, sympy.core.mul.Mul]:
    """Symbolic expressions for the rotated meson to dark matter couplings.

    For the pseudoscalar mediator $\tilde{\\phi}$.
    Comes from `A12<https://arxiv.org/pdf/2611.xxxxx#equation.A.12>`__ and
    `A13<https://arxiv.org/pdf/2611.xxxxx#equation.A.13>`__ in
    `arXiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :param n_dim: Number of dimensions in the rotation.
    :type n_dim: int
    :return: Particle to dark matter rotated couplings, keys are the coupling symbols.
    :rtype: dict[sym.Symbol, sym.core.mul.Mul]
    """
    bare_couplings = sympy.Matrix.zeros(1, n_dim)
    bare_couplings[0] = COUPLING["chi"]
    rotated_couplings = bare_couplings * _create_nd_rotation(n_dim)[0].transpose()
    return dict(
        zip(
            [COUPLING[f"{k}-chi"] for k in ["phi"] + list(PARTICLE_ORDER)],
            rotated_couplings,
        )
    )


# %%
def bare_pseudo_masses() -> dict[sympy.Symbol, sympy.core.mul.Mul]:
    """The off-diagonal bare mass terms.

    For the pseudoscalar mediator $\tilde{\\phi}$.
    Comes from `A5<https://arxiv.org/pdf/2611.xxxxx#equation.A.5>`__ in
    `arXiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :return: Particle-mediator bare masses, keys are the mass symbols.
    :rtype: dict[sympy.Symbol, sympy.core.mul.Mul]
    """
    masses = {}
    masses["pi0"] = (
        (COUPLING["u"] - COUPLING["d"])
        + (COUPLING["u"] - COUPLING["d"])
        * 32
        * PARAMETERS["B0"]
        / PARAMETERS["F"] ** 2
        * PARAMETERS["L6"]
        * (MASS["u"] + MASS["d"] + MASS["s"])
        + 32
        * (COUPLING["u"] + COUPLING["d"] + COUPLING["s"])
        * PARAMETERS["B0"]
        / PARAMETERS["F"] ** 2
        * PARAMETERS["L7"]
        * (MASS["u"] - MASS["d"])
    )
    masses["eta"] = sympy.sqrt(3) / 3 * (
        COUPLING["u"] + COUPLING["d"] - 2 * COUPLING["s"]
    ) * (
        1
        + 32
        * PARAMETERS["B0"]
        / PARAMETERS["F"] ** 2
        * PARAMETERS["L6"]
        * (MASS["u"] + MASS["d"] + MASS["s"])
    ) + 32 * (
        COUPLING["u"] + COUPLING["d"] + COUPLING["s"]
    ) * PARAMETERS[
        "B0"
    ] / PARAMETERS[
        "F"
    ] ** 2 * PARAMETERS[
        "L7"
    ] * (
        MASS["u"] + MASS["d"] - 2 * MASS["s"]
    )
    masses["etaP"] = (
        sympy.sqrt(6)
        / 3
        * (COUPLING["u"] + COUPLING["d"] + COUPLING["s"])
        * (
            1
            + 32
            * PARAMETERS["B0"]
            / PARAMETERS["F"] ** 2
            * (PARAMETERS["L6"] + PARAMETERS["L7"])
            * (MASS["u"] + MASS["d"] + MASS["s"])
        )
    )
    return dict([(MASS[f"{k}-phi"], v) for k, v in masses.items()])


# %%
def rotated_pseudo_photon_coupling(n_dim: int) -> sympy.core.mul.Mul:
    """Symbolic expression for the rotated mediator to photon coupling.

    For the pseudoscalar mediator $\tilde{\\phi}$.
    Comes from `A14<https://arxiv.org/pdf/2611.xxxxx#equation.A.14>`__ and
    `A15<https://arxiv.org/pdf/2611.xxxxx#equation.A.15>`__ in
    `arXiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    Note that the bare un-rotated photon couplings have been eliminated from the
    final expression in favour of the known photon to pion, eta, and eta prime couplings.

    :param n_dim: Number of dimensions in the rotation.
    :type n_dim: int
    :return: Mediator to photon coupling.
    :rtype: sym.core.mul.Mul
    """
    gamma_matrix = (
        sympy.Matrix(
            [[0] + [COUPLING[f"{k}-gamma"] for k in PARTICLE_ORDER[: n_dim - 1]]]
        )
        * _create_nd_rotation(n_dim)[0].transpose()
    )
    gamma_eqns = [
        sympy.Eq(g, e)
        for g, e in zip(
            [
                COUPLING[f"{k}-gamma"]
                for k in ["phi"] + list(PARTICLE_ORDER[: n_dim - 1])
            ],
            gamma_matrix,
        )
    ]
    gamma_solutions = sympy.solve(
        gamma_eqns,
        [COUPLING["phi-gamma"]]
        + [COUPLING[f"{k}Bare-gamma"] for k in PARTICLE_ORDER[: n_dim - 1]],
        dict=True,
    )
    return gamma_solutions[0][COUPLING["phi-gamma"]].trigsimp()


# %%
def bare_pseudo_kaon_couplings() -> dict[sympy.Symbol, sympy.core.mul.Mul]:
    """Symbolic expressions for the un-rotated kaon couplings.

    For the pseudoscalar mediator $\tilde{\\phi}$.
    Implicit in `D16<https://arxiv.org/pdf/2611.xxxxx#equation.D.16>`__ in
    `arXiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :return: Particle to pion-kaon bare couplings, keys are the coupling symbols.
    :rtype: dict[sym.Symbol, sym.core.mul.Mul]
    """
    particle_pi_kaon = {}
    particle_pi_kaon["phi"] = 0 * sympy.Symbol("")
    particle_pi_kaon["pi0"] = -MASS["Kpm"] ** 2 + 2 * MASS["Kpm"] * PARAMETERS["Epipm"]
    particle_pi_kaon["eta"] = (
        MASS["Kpm"] ** 2 - 6 * MASS["Kpm"] * PARAMETERS["Epipm"] + MASS["pipm"] ** 2
    ) / sympy.sqrt(3)
    particle_pi_kaon["etaP"] = (
        sympy.sqrt(6)
        * (3 * PARAMETERS["g8s"] + 2 * PARAMETERS["g8"])
        * (MASS["pipm"] ** 2 - MASS["Kpm"] ** 2)
        / 3
    )
    return dict([(COUPLING[f"{k}Bare-pi-K"], v) for k, v in particle_pi_kaon.items()])


# %%
def rotated_pseudo_kaon_couplings(n_dim: int) -> dict[sympy.Symbol, sympy.core.mul.Mul]:
    """Symbolic expressions for the rotated kaon couplings.

    For the pseudoscalar mediator $\tilde{\\phi}$.
    Comes from `D16<https://arxiv.org/pdf/2611.xxxxx#equation.D.16>`__ in
    `arXiv:2611.xxxxx<https://arxiv.org/abs/2611.xxxxx>`__.

    :param n_dim: Number of dimensions in the rotation.
    :type n_dim: int
    :return:  Particle to pion-kaon bare couplings, keys are the coupling symbols.
    :rtype: dict[sym.Symbol, sym.core.mul.Mul]
    """
    particles = ["phi"] + list(PARTICLE_ORDER[: n_dim - 1])
    bare_couplings = sympy.Matrix([[COUPLING[f"{k}Bare-pi-K"] for k in particles]])
    rotated_couplings = bare_couplings * _create_nd_rotation(n_dim)[0].transpose()
    return dict(zip([COUPLING[f"{k}-pi-K"] for k in particles], rotated_couplings))
