"""Contains functions to numerically evaluate the axial mixing equation."""

# %%
import sympy as sym

# %%
from .symbolic import BARE, PROPAGATION
from .symbolic_axial import kinetic_sines


# %%
def calculate_uvw(
    m_axial: float, m_pi0_axial: float, m_eta_axial: float, m_eta0_axial: float
) -> tuple[float, float, float]:
    """Numerically calculates the kinetic mixing angles u, v, and w.

    For the axial vector mediator case.

    :param m_axial: Mass of axial vector mediator
    :type m_axial: float
    :param m_pi0_axial: Mass of the pi^0
    :type m_pi0_axial: float
    :param m_eta_axial: Mass of the eta
    :type m_eta_axial: float
    :param m_eta0_axial: Mass of the eta^0
    :type m_eta0_axial: float
    :return: u, v, w
    :rtype: tuple[float, float, float]
    """
    parameter = {
        PROPAGATION["A"]: m_axial,
        BARE["pi0-A"]: m_pi0_axial,
        BARE["eta-A"]: m_eta_axial,
        BARE["eta0-A"]: m_eta0_axial,
    }
    uvw = []
    for s in kinetic_sines():
        call = list(s.free_symbols)
        uvw.append(sym.lambdify(call, sym.asin(s))(*[parameter[k] for k in call]))
    return tuple(uvw)
