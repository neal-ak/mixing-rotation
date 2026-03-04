"""Symbolically computes the pseudo scalar mixing equation."""

# %%
import sympy as sym

# %%
from .symbolic import ALL_SYMBOL, create_3d_rotation


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
