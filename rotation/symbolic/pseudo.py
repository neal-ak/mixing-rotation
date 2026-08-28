"""Symbolically computes the pseudo scalar mixing equation."""

# %%
import sympy as sym

# %%
from .general import (
    MASS,
    COUPLING,
    PARAMETERS,
    _create_nd_rotation,
)

PARTICLE_ORDER = "pi0", "eta", "etaP"


# %%
def create_mix_equation(
    rotation: sym.Matrix,
) -> tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix]]:
    """Creates the mixing equation.

    :param rotation: Rotation matrix
    :type rotation: sym.Matrix
    :raises ValueError: Checks if the rotation matrix is square
    :return: Mixing matrix equation, (rotation transpose, bare mass matrix, propagating mass matrix)
    :rtype: tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix]]
    """
    if not rotation.rows == rotation.cols:
        raise ValueError("Rotation matrix must be square")
    particles = ["phi"] + list(PARTICLE_ORDER[: rotation.rows - 1])
    phi_index = particles.index("phi")
    rotation_transpose = rotation.transpose()
    bare_matrix = sym.Matrix.diag([MASS[k + "Bare"] ** 2 for k in particles])
    bare_matrix[:, phi_index] = sym.Matrix(
        [
            MASS[f"{k}-phi"] ** 2 / 2 if k != "phi" else MASS[f"{k}Bare"] ** 2
            for k in particles
        ]
    )
    bare_matrix[phi_index, :] = bare_matrix[:, phi_index].T
    prop_matrix = sym.Matrix.diag([MASS[k] ** 2 for k in particles])
    mixing_equation = sym.Eq(
        prop_matrix,
        sym.MatMul(rotation, bare_matrix, rotation_transpose),
        evaluate=False,
    )
    return mixing_equation, (rotation_transpose, bare_matrix, prop_matrix)


# %%
def rotated_pseudo_meson_couplings(n_dim) -> dict[sym.Symbol, sym.core.mul.Mul]:
    """Symbolic expressions for the rotated meson to dark matter couplings.

    Using the pseudo scalar mediator phi.

    :return:  particle-DM rotated couplings
    :rtype: dict[sympy.Symbol, sympy.core.mul.Mul]
    """
    bare_couplings = sym.Matrix.zeros(1, n_dim)
    bare_couplings[0] = COUPLING["chi"]
    rotated_couplings = bare_couplings * _create_nd_rotation(n_dim)[0].transpose()
    return dict(zip([COUPLING[f"{k}-chi"] for k in PARTICLE_ORDER], rotated_couplings))


# %%
def bare_pseudo_masses() -> dict[sym.Symbol, sym.core.mul.Mul]:
    """The bare pseudo scalar mediator off-diagonal terms.

    :return: particle-mediator bare masses
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
    masses["eta"] = sym.sqrt(3) / 3 * (
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
        sym.sqrt(6)
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
    return dict([(MASS[f"{k}-phi"], masses[k]) for k in PARTICLE_ORDER])


# %%
def rotated_pseudo_photon_coupling(n_dim) -> sym.core.mul.Mul:
    """Symbolic expression for the rotated mediator to photon coupling.

    Using the pseudo scalar mediator phi.

    Note that the bare un-rotated photon couplings have been eliminated from the
    final expression in favour of the known photon to pion, eta, and eta prime couplings.

    :return: mediator-photon
    :rtype: sym.core.mul.Mul
    """
    gamma_matrix = (
        sym.Matrix(
            [[0] + [COUPLING[f"{k}-gamma"] for k in PARTICLE_ORDER[: n_dim - 1]]]
        )
        * _create_nd_rotation(n_dim)[0].transpose()
    )
    gamma_eqns = [
        sym.Eq(g, e)
        for g, e in zip(
            [
                COUPLING[f"{k}-gamma"]
                for k in ["phi"] + list(PARTICLE_ORDER[: n_dim - 1])
            ],
            gamma_matrix,
        )
    ]
    gamma_solutions = sym.solve(
        gamma_eqns,
        [COUPLING["phi-gamma"]]
        + [COUPLING[f"{k}Bare-gamma"] for k in PARTICLE_ORDER[: n_dim - 1]],
        dict=True,
    )
    return gamma_solutions[0][COUPLING["phi-gamma"]].trigsimp()


# %%
def bare_pseudo_kaon_couplings() -> dict[sym.Symbol, sym.core.mul.Mul]:
    """Symbolic expressions for the un-rotated kaon couplings.

    Using the pseudo scalar mediator phi.

    :return: particle-pion-kaon bare couplings
    :rtype: dict[sympy.Symbol, sympy.core.mul.Mul]
    """
    particle_pi_kaon = {}
    particle_pi_kaon["phi"] = 0 * sym.Symbol("a")
    particle_pi_kaon["pi0"] = -MASS["Kpm"] ** 2 + 2 * MASS["Kpm"] * PARAMETERS["Epipm"]
    particle_pi_kaon["eta"] = (
        MASS["Kpm"] ** 2 - 6 * MASS["Kpm"] * PARAMETERS["Epipm"] + MASS["pipm"] ** 2
    ) / sym.sqrt(3)
    particle_pi_kaon["etaP"] = (
        sym.sqrt(6)
        * (3 * PARAMETERS["g8s"] + 2 * PARAMETERS["g8"])
        * (MASS["pipm"] ** 2 - MASS["Kpm"] ** 2)
        / 3
    )
    return dict([(COUPLING[f"{k}Bare-pi-K"], particle_pi_kaon[k]) for k in particles])


# %%
def rotated_pseudo_kaon_couplings(n_dim) -> dict[sym.Symbol, sym.core.mul.Mul]:
    """Symbolic expressions for the rotated kaon couplings.

    Using the pseudo scalar mediator phi.

    :return:  particle-pi-kaon rotated couplings
    :rtype: dict[sympy.Symbol, sympy.core.mul.Mul]
    """
    particles = ["phi"] + list(PARTICLE_ORDER[: n_dim - 1])
    bare_couplings = sym.Matrix([[COUPLING[f"{k}Bare-pi-K"] for k in particles]])
    rotated_couplings = bare_couplings * _create_nd_rotation(n_dim)[0].transpose()
    return dict(zip([COUPLING[f"{k}-pi-K"] for k in particles], rotated_couplings))
