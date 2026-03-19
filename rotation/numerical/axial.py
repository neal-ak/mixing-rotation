"""Contains functions to numerically evaluate the axial mixing equation."""

# %%
import sympy as sym

# %%
from ..symbolic.general import (
    BARE,
    PROPAGATION,
    ANGLE,
    AXIAL_COUPLING,
    PARAMETERS,
)
from ..symbolic.axial import (
    kinetic_sines,
    rotated_axial_meson_couplings,
    bare_axial_masses,
    bare_axial_kaon_couplings,
    rotated_axial_kaon_couplings,
)
from .solver import PION_DECAY, MASS_QUARK, MASS_CHARGED_PION, MASS_CHARGED_KAON


# %%
def calculate_uvw(
    m_axial: float, m_pi0_axial: float, m_eta_axial: float, m_etap_axial: float
) -> tuple[float, float, float]:
    """Numerically calculates the kinetic mixing angles u, v, and w.

    For the axial vector mediator case.

    :param m_axial: Mass of axial vector mediator
    :type m_axial: float
    :param m_pi0_axial: Neutral pion - mediator mass
    :type m_pi0_axial: float
    :param m_eta_axial: Eta - mediator mass
    :type m_eta_axial: float
    :param m_etap_axial: Eta prime - mediator mass
    :type m_etap_axial: float
    :return: u, v, w
    :rtype: tuple[float, float, float]
    """
    parameter = {
        PROPAGATION["AP"]: m_axial,
        BARE["pi0-A"]: m_pi0_axial,
        BARE["eta-A"]: m_eta_axial,
        BARE["etaP-A"]: m_etap_axial,
    }
    uvw = []
    for s in kinetic_sines():
        call = list(s.free_symbols)
        uvw.append(sym.lambdify(call, sym.asin(s))(*[parameter[k] for k in call]))
    return tuple(uvw)


def axial_calculate_meson_couplings(
    l_axial: float,
    m_axial: float,
    m_pi0_axial: float,
    m_eta_axial: float,
    m_etap_axial: float,
    u: float,
    v: float,
    w: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Numerically calculates the rotated meson to dark matter couplings.

    Specific to the axial vector mediator case.

    :param l_axial: Axial vector mediator coupling
    :type l_axial: float
    :param m_axial: Axial vector mediator mass
    :type m_axial: float
    :param m_pi0_axial: Neutral pion - mediator mass
    :type m_pi0_axial: float
    :param m_eta_axial: Eta - mediator mass
    :type m_eta_axial: float
    :param m_etap_axial: Eta prime - mediator mass
    :type m_etap_axial: float
    :param u: 1st kinetic rotation angle
    :type u: float
    :param v: 2nd kinetic rotation angle
    :type v: float
    :param w: 3rd kinetic rotation angle
    :type w: float
    :param alpha: 1st rotation angle
    :type alpha: float
    :param beta: 2nd rotation angle
    :type beta: float
    :param gamma: 3rd rotation angle
    :type gamma: float
    :return: pion-chi, eta-chi, eta^prime-chi
    :rtype: tuple[float, float, float]
    """
    parameter = {
        AXIAL_COUPLING["a"]: l_axial,
        BARE["A"]: m_axial,
        BARE["pi0-A"]: m_pi0_axial,
        BARE["eta-A"]: m_eta_axial,
        BARE["etaP-A"]: m_etap_axial,
        ANGLE["u"]: u,
        ANGLE["v"]: v,
        ANGLE["w"]: w,
        ANGLE["alpha"]: alpha,
        ANGLE["beta"]: beta,
        ANGLE["gamma"]: gamma,
    }
    coupling = []
    for l in rotated_axial_meson_couplings():
        call = list(l.free_symbols)
        coupling.append(sym.lambdify(call, sym.asin(l))(*[parameter[k] for k in call]))
    return tuple(coupling)


def axial_calculate_bare_masses(
    b_0: float,
    l_4: float,
    l_5: float,
    coupling_up: float,
    coupling_down: float,
    coupling_strange: float,
    pion_decay: float = PION_DECAY,
    mass_up: float = MASS_QUARK["up"],
    mass_down: float = MASS_QUARK["down"],
    mass_strange: float = MASS_QUARK["strange"],
) -> tuple[float, float, float]:
    """Numerical calculation of the bare off-diagonal meson masses.

    Specific to the axial vector mediator case.

    :param b_0: B_{0}
    :type b_0: float
    :param l_4: L_{4}
    :type l_4: float
    :param l_5: L_{5}
    :type l_5: float
    :param coupling_up: Up coupling
    :type coupling_up: float
    :param coupling_down: Down coupling
    :type coupling_down: float
    :param coupling_strange: Strange coupling
    :type coupling_strange: float
    :param pion_decay: The pion decay constant, defaults to PION_DECAY
    :type pion_decay: float
    :param mass_up: Up mass, defaults to MASS_QUARK["up"]
    :type mass_up: float, optional
    :param mass_down: Down mass, defaults to MASS_QUARK["down"]
    :type mass_down: float, optional
    :param mass_strange: Strange mass, defaults to MASS_QUARK["strange"]
    :type mass_strange: float, optional
    :return: pion-mediator, eta-mediator, eta^prime-mediator
    :rtype: tuple[float, float, float]
    """
    parameter = {
        PARAMETERS["B0"]: b_0,
        PARAMETERS["F"]: pion_decay,
        PARAMETERS["L4"]: l_4,
        PARAMETERS["L5"]: l_5,
        PARAMETERS["mu"]: mass_up,
        PARAMETERS["md"]: mass_down,
        PARAMETERS["ms"]: mass_strange,
        PARAMETERS["au"]: coupling_up,
        PARAMETERS["ad"]: coupling_down,
        PARAMETERS["as"]: coupling_strange,
    }
    masses = []
    for m in bare_axial_masses():
        call = list(m.free_symbols)
        masses.append(sym.lambdify(call, m)(*[parameter[k] for k in call]))
    return tuple(masses)


def axial_calculate_bare_kaon(
    energy_pion: float,
    energy_dm: float,
    mass_axial: float,
    g_8: float,
    g_8s: float,
    coupling_up: float,
    coupling_down: float,
    coupling_strange: float,
    mass_pion_axial: float,
    mass_eta_axial: float,
    mass_etap_axial: float,
    pion_decay: float = PION_DECAY,
    mass_pion: float = MASS_CHARGED_PION,
    mass_kaon: float = MASS_CHARGED_KAON,
) -> tuple[float, float, float]:
    """Numerical calculation of the un-rotated kaon couplings.

    Specific to the axial vector mediator case. Defaults are from the PDG
    [charged pion](https://pdglive.lbl.gov/Particle.action?node=S008) and
    [charged kaon](https://pdglive.lbl.gov/Particle.action?node=S010).

    :param energy_pion: Pion energy
    :type energy_pion: float
    :param energy_dm: Dark matter energy
    :type energy_dm: float
    :param mass_axial: Axial vector mediator mass
    :type mass_axial: float
    :param g_8: g_{8}
    :type g_8: float
    :param g_8s: g_{8s}
    :type g_8s: float
    :param coupling_up: Up coupling
    :type coupling_up: float
    :param coupling_down: Down coupling
    :type coupling_down: float
    :param coupling_strange: Strange coupling
    :type coupling_strange: float
    :param mass_pion_axial: Pion - mediator mass
    :type mass_pion_axial: float
    :param mass_eta_axial: Eta - mediator mass
    :type mass_eta_axial: float
    :param mass_etap_axial: Eta prime - mediator mass
    :type mass_etap_axial: float
    :param pion_decay: The pion decay constant, defaults to PION_DECAY
    :type pion_decay: float
    :param mass_pion: Chared pion mass, defaults to MASS_CHARGED_PION
    :type mass_pion: float, optional
    :param mass_kaon: Charged kaon mass, defaults to MASS_CHARGED_KAON
    :type mass_kaon: float, optional
    :return: pion-pion-kaon, eta-pion-kaon, eta^prime-pion-kaon
    :rtype: tuple[float, float, float]
    """
    parameter = {
        PARAMETERS["Epipm"]: energy_pion,
        PARAMETERS["Echi"]: energy_dm,
        BARE["A"]: mass_axial,
        PARAMETERS["F"]: pion_decay,
        PARAMETERS["g8"]: g_8,
        PARAMETERS["g8s"]: g_8s,
        PARAMETERS["au"]: coupling_up,
        PARAMETERS["ad"]: coupling_down,
        PARAMETERS["as"]: coupling_strange,
        BARE["pi0-A"]: mass_pion_axial,
        BARE["eta-A"]: mass_eta_axial,
        BARE["etaP-A"]: mass_etap_axial,
        PROPAGATION["pipm"]: mass_pion,
        PROPAGATION["Kpm"]: mass_kaon,
    }
    coupling = []
    for l in bare_axial_kaon_couplings():
        call = list(l.free_symbols)
        coupling.append(sym.lambdify(call, l)(*[parameter[k] for k in call]))
    return tuple([coupling[0], 0, coupling[1]])


def axial_calculate_kaon_couplings(
    bare_pion_pion_kaon: float,
    bare_eta_pion_kaon: float,
    bare_etap_pion_kaon: float,
    u: float,
    v: float,
    w: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Numerical calculation of the rotated kaon couplings.

    Specific to the axial vector mediator case.

    :param bare_pion_pion_kaon: Un-rotated pion-pion-kaon coupling
    :type bare_pion_pion_kaon: float
    :param bare_eta_pion_kaon: Un-rotated eta-pion-kaon coupling
    :type bare_eta_pion_kaon: float
    :param bare_etap_pion_kaon: Un-rotated eta^prime-pion-kaon coupling
    :type bare_etap_pion_kaon: float
    :param u: 1st kinetic rotation angle
    :type u: float
    :param v: 2nd kinetic rotation angle
    :type v: float
    :param w: 3rd kinetic rotation angle
    :type w: float
    :param alpha: 1st rotation angle
    :type alpha: float
    :param beta: 2nd rotation angle
    :type beta: float
    :param gamma: 3rd rotation angle
    :type gamma: float
    :return: pion-pion-kaon, eta-pion-kaon, eta^prime-pion-kaon
    :rtype: tuple[float, float, float]
    """
    parameter = {
        AXIAL_COUPLING["pi0Bare-pi-K"]: bare_pion_pion_kaon,
        AXIAL_COUPLING["etaBare-pi-K"]: bare_eta_pion_kaon,
        AXIAL_COUPLING["etaPBare-pi-K"]: bare_etap_pion_kaon,
        ANGLE["u"]: u,
        ANGLE["v"]: v,
        ANGLE["w"]: w,
        ANGLE["alpha"]: alpha,
        ANGLE["beta"]: beta,
        ANGLE["gamma"]: gamma,
    }
    coupling = []
    for g in rotated_axial_kaon_couplings():
        call = list(g.free_symbols)
        coupling.append(sym.lambdify(call, g)(*[parameter[k] for k in call]))
    return tuple(coupling)
