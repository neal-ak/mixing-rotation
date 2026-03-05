"""Symbolically computes the pseudo scalar mixing equation."""

# %%
import sympy as sym

# %%
from .general import (
    BARE,
    COUPLING,
    PARAMETERS,
    PROPAGATION,
    ROTATED,
    create_3d_rotation,
)


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
            [BARE["pi0"] ** 2, BARE["pi0-phi"] / 2, 0],
            [
                BARE["pi0-phi"] / 2,
                BARE["phi"] ** 2,
                BARE["eta-phi"] / 2,
            ],
            [0, BARE["eta-phi"] / 2, BARE["eta"] ** 2],
        ]
    )
    prop_matrix = sym.Matrix(
        [
            [PROPAGATION["pi0"] ** 2, 0, 0],
            [0, PROPAGATION["phi"] ** 2, 0],
            [0, 0, PROPAGATION["eta"] ** 2],
        ]
    )
    mixing_equation = sym.Eq(
        prop_matrix,
        sym.MatMul(rotation, bare_matrix, rotation_transpose),
        evaluate=False,
    )
    return mixing_equation, (rotation_transpose, bare_matrix, prop_matrix)


# %%
def rotated_pseudo_meson_couplings() -> (
    tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
):
    """Symbolic expressions for the rotated meson to dark matter couplings.

    Using the pseudo scalar mediator phi.

    :return: pion-chi, mediator-chi, eta-chi
    :rtype: tuple[sympy.core.mul.Mul, sympy.core.mul.Mul, sympy.core.mul.Mul]
    """
    bare_couplings = sym.Matrix([[0, COUPLING["phi-chi"], 0]])
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
                    COUPLING["pi-gamma"],
                    0,
                    COUPLING["eta-gamma"],
                ]
            ]
        )
        * create_3d_rotation()[0].transpose()
    )
    gamma_eqns = [
        sym.Eq(g, e)
        for g, e in zip(
            [ROTATED[k] for k in ["pi-gamma", "phi-gamma", "eta-gamma"]],
            gamma_matrix,
        )
    ]
    gamma_solutions = sym.solve(
        gamma_eqns,
        [
            ROTATED["phi-gamma"],
            COUPLING["pi-gamma"],
            COUPLING["eta-gamma"],
        ],
        dict=True,
    )
    return gamma_solutions[0][ROTATED["phi-gamma"]].trigsimp()


# %%
def bare_pseudo_kaon_couplings() -> (
    tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
):
    """Symbolic expressions for the un-rotated kaon couplings.

    Using the pseudo scalar mediator phi.

    :return: pion-pion-kaon, eta-pion-kaon
    :rtype: tuple[sym.core.mul.Mul, sym.core.mul.Mul]
    """
    pi_pi_kaon = PROPAGATION["K0"] ** 2 - 2 * PROPAGATION["K0"] * PARAMETERS["Epi"]
    eta_pi_kaon = (
        PROPAGATION["K0"] ** 2
        - 2 * PROPAGATION["K0"] * PARAMETERS["Epi"]
        - PROPAGATION["pi0"] ** 2
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
    bare_couplings = sym.Matrix([[COUPLING[f"{k}-pi-K"] for k in ["pi", "phi", "eta"]]])
    rotated_couplings = bare_couplings * create_3d_rotation()[0].transpose()
    return tuple(rotated_couplings)
