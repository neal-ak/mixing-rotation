"""Contains functions to numerically evaluate the pseudo scalar mixing equation."""

# %%
import sympy as sym

# %%
from ..symbolic.general import (
    ANGLE,
    COUPLING,
    PARAMETERS,
    MASS,
)
from ..symbolic.pseudo import (
    rotated_pseudo_meson_couplings,
    bare_pseudo_masses,
    rotated_pseudo_photon_coupling,
    bare_pseudo_kaon_couplings,
    rotated_pseudo_kaon_couplings,
)
from .solver import (
    PION_DECAY,
    MASS_QUARK,
    MASS_CHARGED_PION,
    MASS_CHARGED_KAON,
    B_0,
    L_6,
    L_7,
    G_8,
    G_8S,
    G_PI_GAMMA,
    G_ETA_GAMMA,
    G_ETA_PRIME_GAMMA,
)


# %%
def pseudo_calculate_meson_couplings(
    n_dim: int,
    bare_mediator_dm_coupling: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict[sympy.Symbol, float]:
    """Numerical calculation of the rotated meson to dark matter couplings.

    Specific to the pseudo scalar mediator phi case.

    :param n_dim: number of rotation dimension
    :type n_dim: int
    :param bare_mediator_dm_coupling: un-rotated mediator to dark matter coupling
    :type bare_mediator_dm_coupling: float
    :param alpha: mixing angle alpha
    :type alpha: float
    :param beta: mixing angle beta
    :type beta: float
    :param gamma: mixing angle gamma
    :type gamma: float
    :return: rotated particle-DM couplings
    :rtype: dict[sympy.Symbol, float]
    """
    parameter = {
        COUPLING["chi"]: bare_mediator_dm_coupling,
        ANGLE["alpha"]: alpha,
        ANGLE["beta"]: beta,
        ANGLE["gamma"]: gamma,
    }
    coupling = {}
    for k, g in rotated_pseudo_meson_couplings(n_dim).items():
        call = list(g.free_symbols)
        coupling[k] = sym.lambdify(call, g)(*[parameter[k] for k in call])
    return coupling


def pseudo_calculate_bare_masses(
    coupling_up: float,
    coupling_down: float,
    coupling_strange: float,
    b_0: float = B_0,
    pion_decay: float = PION_DECAY,
) -> dict[sym.Symbol, float]:
    """Numerical calculation of the bare off-diagonal meson masses.

    Specific to the pseudo scalar mediator phi case.

    :param coupling_up: Up coupling
    :type coupling_up: float
    :param coupling_down: Down coupling
    :type coupling_down: float
    :param coupling_strange: Strange coupling
    :type coupling_strange: float
    :param b_0: B_{0}, defaults to B_0
    :type b_0: float
    :param pion_decay: The pion decay constant, defaults to PION_DECAY
    :type pion_decay: float
    :return: bare particle-mediator couplings
    :rtype: dict[sympy.Symbol, float]
    """
    parameter = {
        COUPLING["u"]: coupling_up,
        COUPLING["d"]: coupling_down,
        COUPLING["s"]: coupling_strange,
        MASS["u"]: MASS_QUARK["up"],
        MASS["d"]: MASS_QUARK["down"],
        MASS["s"]: MASS_QUARK["strange"],
        PARAMETERS["B0"]: b_0,
        PARAMETERS["F"]: pion_decay,
        PARAMETERS["L6"]: L_6,
        PARAMETERS["L7"]: L_7,
    }
    masses = {}
    for k, m in bare_pseudo_masses().items():
        call = list(m.free_symbols)
        masses[k] = sym.lambdify(call, m)(*[parameter[k] for k in call])
    return masses


# %%
def pseudo_calculate_photon_coupling(
    n_dim: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float = None,
    zeta: float = None,
    kappa: float = None,
    known_pion_photon_coupling: float = G_PI_GAMMA,
    known_eta_photon_coupling: float = G_ETA_GAMMA,
    known_etaP_photon_coupling: float = G_ETA_PRIME_GAMMA,
) -> float:
    """Numerical calculation of the rotated mediator to photon coupling.

    Specific to the pseudo scalar mediator case.

    :param n_dim: number of rotation dimension
    :type n_dim: int
    :param alpha: alpha mixing angle
    :type alpha: float
    :param beta: beta mixing angle
    :type beta: float
    :param gamma: gamma mixing angle
    :type gamma: float
    :param delta: delta mixing angle, defaults to None
    :type delta: float
    :param zeta: zeta mixing angle, defaults to None
    :type zeta: float
    :param kappa: kappa mixing angle, defaults to None
    :type kappa: float
    :param known_pion_photon_coupling: known pion to photon coupling, defaults to G_PI_GAMMA
    :type known_pion_photon_coupling: float
    :param known_eta_photon_coupling: known eta to photon coupling, defaults to G_ETA_GAMMA
    :type known_eta_photon_coupling: float
    :param known_etaP_photon_coupling: known eta prime to photon coupling, defaults to G_ETA_PRIME_GAMMA
    :type known_etaP_photon_coupling: float
    :return: rotated mediator to photon coupling
    :rtype: float
    """
    parameter = {
        ANGLE["alpha"]: alpha,
        ANGLE["beta"]: beta,
        ANGLE["gamma"]: gamma,
        ANGLE["delta"]: delta,
        ANGLE["zeta"]: zeta,
        ANGLE["kappa"]: kappa,
        COUPLING["pi0-gamma"]: known_pion_photon_coupling,
        COUPLING["eta-gamma"]: known_eta_photon_coupling,
        COUPLING["etaP-gamma"]: known_etaP_photon_coupling,
    }
    g = rotated_pseudo_photon_coupling(n_dim)
    call = list(g.free_symbols)
    return sym.lambdify(call, g)(*[parameter[k] for k in call])


# %%
def pseudo_calculate_bare_kaon(
    energy_pion: float,
    energy_kaon: float,
    mass_pion: float = MASS_CHARGED_PION,
    mass_kaon: float = MASS_CHARGED_KAON,
) -> dict[sym.Symbol, float]:
    """Numerical calculation of the un-rotated kaon couplings.

    Specific to the pseudo scalar mediator case.

    :param energy_pion: Pion energy
    :type energy_pion: float
    :param energy_kaon: Kaon energy
    :type energy_kaon: float
    :param mass_pion: Charged pion mass, defaults to MASS_CHARGED_PION
    :type mass_pion: float
    :param mass_kaon: Charged kaon mass, defaults to MASS_CHARGED_KAON
    :type mass_kaon: float
    :return: bare particle-pion-kaon couplings
    :rtype: dict[sympy.Symbol, float]
    """
    parameter = {
        PARAMETERS["Epipm"]: energy_pion,
        PARAMETERS["EKpm"]: energy_kaon,
        PARAMETERS["g8"]: G_8,
        PARAMETERS["g8s"]: G_8S,
        MASS["pipm"]: mass_pion,
        MASS["Kpm"]: mass_kaon,
    }
    coupling = {}
    for s, g in bare_pseudo_kaon_couplings().items():
        call = list(g.free_symbols)
        coupling[s] = sym.lambdify(call, g)(*[parameter[k] for k in call])
    return coupling


# %%
def pseudo_calculate_kaon_couplings(
    n_dim: int,
    bare_pion_pion_kaon_coupling: float,
    bare_mediator_pion_kaon_coupling: float,
    bare_eta_pion_kaon_coupling: float,
    bare_etaP_pion_kaon_coupling: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float = None,
    zeta: float = None,
    kappa: float = None,
):
    """Numerical calculation of the rotated kaon couplings.

    Specific to the pseudo scalar mediator case.

    :param n_dim: number of rotation dimension
    :type n_dim: int
    :param bare_pion_pion_kaon_coupling: Un-rotated pion-pion-kaon coupling
    :type bare_pion_pion_kaon_coupling: float
    :param bare_mediator_pion_kaon_coupling: Un-rotated mediator-pion-kaon coupling
    :type bare_mediator_pion_kaon_coupling: float
    :param bare_eta_pion_kaon_coupling: Un-rotated eta-pion-kaon coupling
    :type bare_eta_pion_kaon_coupling: float
    :param bare_etaP_pion_kaon_coupling: Un-rotated eta prime-pion-kaon coupling
    :type bare_etaP_pion_kaon_coupling: float
    :param alpha: Alpha mixing angle
    :type alpha: float
    :param beta: Beta mixing angle
    :type beta: float
    :param gamma: Gamma mixing
    :type gamma: float
    :param delta: delta mixing angle, defaults to None
    :type delta: float
    :param zeta: zeta mixing angle, defaults to None
    :type zeta: float
    :param kappa: kappa mixing angle, defaults to None
    :type kappa: float
    :return: rotated particle-pion-kaon couplings
    :rtype: _type_
    """
    parameter = {
        COUPLING["pi0Bare-pi-K"]: bare_pion_pion_kaon_coupling,
        COUPLING["phiBare-pi-K"]: bare_mediator_pion_kaon_coupling,
        COUPLING["etaBare-pi-K"]: bare_eta_pion_kaon_coupling,
        COUPLING["etaPBare-pi-K"]: bare_etaP_pion_kaon_coupling,
        ANGLE["alpha"]: alpha,
        ANGLE["beta"]: beta,
        ANGLE["gamma"]: gamma,
        ANGLE["delta"]: delta,
        ANGLE["zeta"]: zeta,
        ANGLE["kappa"]: kappa,
    }
    coupling = {}
    for s, g in rotated_pseudo_kaon_couplings(n_dim).items():
        call = list(g.free_symbols)
        coupling[s] = sym.lambdify(call, g)(*[parameter[k] for k in call])
    return coupling
