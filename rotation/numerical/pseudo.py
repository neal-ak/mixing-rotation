"""Contains functions to numerically evaluate the pseudo scalar mixing equation."""

# %%
import sympy as sym

# %%
from ..symbolic.symbolic import ANGLE, COUPLING, PARAMETERS, PROPAGATION, ROTATED
from ..symbolic.symbolic_pseudo import (
    rotated_pseudo_meson_couplings,
    rotated_pseudo_photon_coupling,
    bare_pseudo_kaon_couplings,
    rotated_pseudo_kaon_couplings,
)


# %%
def pseudo_calculate_meson_couplings(
    bare_mediator_dm_coupling: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Numerical calculation of the rotated meson to dark matter couplings.

    Specific to the pseudo scalar mediator phi case.

    :param bare_mediator_dm_coupling: un-rotated mediator to dark matter coupling
    :type bare_mediator_dm_coupling: float
    :param alpha: mixing angle alpha
    :type alpha: float
    :param beta: mixing angle beta
    :type beta: float
    :param gamma: mixing angle gamma
    :type gamma: float
    :return: pion-chi, phi-chi, eta-chi
    :rtype: tuple[float, float, float]
    """
    parameter = {
        COUPLING["phi-chi"]: bare_mediator_dm_coupling,
        ANGLE["alpha"]: alpha,
        ANGLE["beta"]: beta,
        ANGLE["gamma"]: gamma,
    }
    coupling = []
    for g in rotated_pseudo_meson_couplings():
        call = list(g.free_symbols)
        coupling.append(sym.lambdify(call, g)(*[parameter[k] for k in call]))
    return tuple(coupling)


# %%
def pseudo_calculate_photon_coupling(
    known_pion_photon_coupling: float,
    known_eta_photon_coupling: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Numerical calculation of the rotated mediator to photon coupling.

    Specific to the pseudo scalar mediator case.

    :param known_pion_photon_coupling: known pion to photon coupling
    :type known_pion_photon_coupling: float
    :param known_eta_photon_coupling: known eta to photon coupling
    :type known_eta_photon_coupling: float
    :param alpha: alpha mixing angle
    :type alpha: float
    :param beta: beta mixing angle
    :type beta: float
    :param gamma: gamma mixing angle
    :type gamma: float
    :return: rotated mediator to photon coupling
    :rtype: float
    """
    parameter = {
        ROTATED["pi-gamma"]: known_pion_photon_coupling,
        ROTATED["eta-gamma"]: known_eta_photon_coupling,
        ANGLE["alpha"]: alpha,
        ANGLE["beta"]: beta,
        ANGLE["gamma"]: gamma,
    }
    g = rotated_pseudo_photon_coupling()
    call = list(g.free_symbols)
    return sym.lambdify(call, g)(*[parameter[k] for k in call])


# %%
def pseudo_calculate_bare_kaon(
    energy_pion: float,
    mass_pion: float,
    mass_kaon: float,
) -> tuple[float, float, float]:
    """Numerical calculation of the un-rotated kaon couplings.

    Specific to the pseudo scalar mediator case.

    :param energy_pion: Pion energy
    :type energy_pion: float
    :param mass_pion: Pion mass
    :type mass_pion: float
    :param mass_kaon: Kaon mass
    :type mass_kaon: float
    :return: pion-pion-kaon, mediator-pion-kaon, eta-pion-kaon
    :rtype: tuple[float, float, float]
    """
    parameter = {
        PARAMETERS["Epi"]: energy_pion,
        PROPAGATION["pi0"]: mass_pion,
        PROPAGATION["K0"]: mass_kaon,
    }
    coupling = []
    for g in bare_pseudo_kaon_couplings():
        call = list(g.free_symbols)
        coupling.append(sym.lambdify(call, g)(*[parameter[k] for k in call]))
    return tuple([coupling[0], 0, coupling[1]])


# %%
def pseudo_calculate_kaon_couplings(
    bare_pion_pion_kaon_coupling: float,
    bare_mediator_pion_kaon_coupling: float,
    bare_eta_pion_kaon_coupling: float,
    alpha: float,
    beta: float,
    gamma: float,
):
    """Numerical calculation of the rotated kaon couplings.

    Specific to the pseudo scalar mediator case.

    :param bare_pion_pion_kaon_coupling: Un-rotated pion-pion-kaon coupling
    :type bare_pion_pion_kaon_coupling: float
    :param bare_mediator_pion_kaon_coupling: Un-rotated mediator-pion-kaon coupling
    :type bare_mediator_pion_kaon_coupling: float
    :param bare_eta_pion_kaon_coupling: Un-rotated eta-pion-kaon coupling
    :type bare_eta_pion_kaon_coupling: float
    :param alpha: Alpha mixing angle
    :type alpha: float
    :param beta: Beta mixing angle
    :type beta: float
    :param gamma: Gamma mixing
    :type gamma: float
    :return: rotated pion-pion-kaon, mediator-pion-kaon, eta-pion-kaon
    :rtype: _type_
    """
    parameter = {
        COUPLING["pi-pi-K"]: bare_pion_pion_kaon_coupling,
        COUPLING["phi-pi-K"]: bare_mediator_pion_kaon_coupling,
        COUPLING["eta-pi-K"]: bare_eta_pion_kaon_coupling,
        ANGLE["alpha"]: alpha,
        ANGLE["beta"]: beta,
        ANGLE["gamma"]: gamma,
    }
    coupling = []
    for g in rotated_pseudo_kaon_couplings():
        call = list(g.free_symbols)
        coupling.append(sym.lambdify(call, g)(*[parameter[k] for k in call]))
    return tuple(coupling)
