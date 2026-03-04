"""Contains the numerical solver for the mixing equation."""

# %%
from collections.abc import Callable

# %%
import numpy as np
from scipy import optimize
import sympy as sym

# %%
from .symbolic import (
    ALL_SYMBOL,
    rotated_pseudo_meson_couplings,
    rotated_pseudo_photon_coupling,
    bare_pseudo_kaon_couplings,
    rotated_pseudo_kaon_couplings,
)
from .symbolic_axial import kinetic_sines

# %%
_DEFAULT_BOUNDS = {}
for section, sub_symbols in ALL_SYMBOL.items():
    match section:
        case "prop" | "bare":
            _DEFAULT_BOUNDS |= {v: tuple([0, np.inf]) for v in sub_symbols.values()}
        case "angle":
            _DEFAULT_BOUNDS |= {v: tuple([-np.pi, np.pi]) for v in sub_symbols.values()}
        case "cos" | "sin":
            _DEFAULT_BOUNDS |= {v: tuple([-1, 1]) for v in sub_symbols.values()}
        case "tan":
            _DEFAULT_BOUNDS |= {
                v: tuple([-np.inf, np.inf]) for v in sub_symbols.values()
            }
        case _:
            _DEFAULT_BOUNDS |= {
                v: tuple([-np.inf, np.inf]) for v in sub_symbols.values()
            }


# %%
def _build_call(
    arg_ordering: list[sym.Symbol],
    x0: list[float],
    knowns: dict[sym.Symbol, float],
) -> list[float]:
    """Create call signature for lambdified Sympy functions.

    :param arg_ordering: Call order defined by the Sympy lambdification
    :type arg_ordering: list[sympy.Symbol]
    :param x0: Values for unknown parameters
    :type x0: list[float]
    :param knowns: Known arguments and their fixed values
    :type knowns: dict[sympy.Symbol, float]
    :return: Call signature
    :rtype: list[float]
    """
    x0 = iter(x0)
    call = {}
    # A. Pull from the knowns
    # B. If that fails, get the next unknown that was passed in
    for argument in arg_ordering:
        if argument in knowns:
            call[argument] = knowns[argument]
        else:
            call[argument] = next(x0)

    return [call[argument] for argument in arg_ordering]


# %%
def _least_sqr_tolerance(
    function: Callable,
    arg_ordering: list[sym.Symbol],
    init_unknowns: dict[sym.Symbol, float],
    abs_tolerance: float | None,
    **kwargs,
) -> tuple[dict[sym.Symbol, float], optimize.OptimizeResult]:
    """Runs least squares optimization and checks solution for validity.

    The tolerance check can be disabled by setting the tolerance to None.

    :param function: Function that least squares will optimize
    :type function: Callable
    :param arg_ordering: Call order defined by the Sympy lambdification
    :type arg_ordering: list[sympy.Symbol]
    :param init_unknowns: Unknown arguments and their initial guesses
    :type init_unknowns: dict[sympy.Symbol, float]
    :param abs_tolerance: Absolute tolerance for the check, disable with None
    :type abs_tolerance: float | None
    :raises ValueError: Raised if solution outside tolerance
    :return: Dictionary of solved values and the raw OptimizeResult output
    :rtype: tuple[dict[sympy.Symbol, float], scipy.optimize.OptimizeResult]
    """
    unknown_args = [k for k in arg_ordering if k in init_unknowns]
    solution = optimize.least_squares(
        function,
        [init_unknowns[k] for k in unknown_args],
        **{k: v for k, v in kwargs.items() if k not in ["fun", "x0"]},
    )
    solve = dict(zip(unknown_args, solution.x))

    if abs_tolerance is not None:
        correct = np.isclose(
            function([solve[k] for k in arg_ordering if k in solve]),
            0,
            atol=abs_tolerance,
        )
        if not np.all(correct):
            raise ValueError("Solutions not close enough to zero")
    return solve, solution


# %%
def initialize_system(equation: sym.Matrix) -> tuple[list[sym.Symbol], list[Callable]]:
    """Initializes a system of equations for the solver.

    Sets up the argument call order for the lambdified sympy equations, the
    equations themselves and their partials. Due to redundancy the callable
    system drops the lower diagonal component of the matrix (6 equations in the
    system).

    :param equation: Matrix representing the system of equations
    :type equation: sym.Matrix
    :return: Argument call order, and the system of equations
    :rtype: tuple[list[sympy.Symbol], list[Callable]]
    """
    arg_ordering = []
    for s in ALL_SYMBOL.values():
        arg_ordering += [v for v in s.values() if v in equation.free_symbols]
    system = []
    for eqn in [e for i, e in enumerate(equation) if i not in [3, 6, 7]]:
        system.append(sym.lambdify(arg_ordering, eqn))
    return arg_ordering, system


# %%
def create_func(
    system: list[Callable],
    arg_ordering: list[sym.Symbol],
    knowns: dict[sym.Symbol, float],
) -> Callable:
    """Creates the vector function for the Scipy solver.

    :param system: System of equations lambdified from Sympy
    :type system: list[Callable]
    :param arg_ordering: Call order defined by the Sympy lambdification
    :type arg_ordering: list[sympy.Symbol]
    :param arg_unknown: The unknowns as a subset of the call order
    :type arg_unknown: list[sympy.Symbol]
    :param knowns: Known arguments and their values in the system
    :type knowns: dict[sympy.Symbol, float]
    :return: The function Scipy can use to optimize
    :rtype: Callable
    """

    def function(x):
        calling = _build_call(arg_ordering, x, knowns)
        return [f(*calling) for f in system]

    return function


# %%
def debug_solve_mixing(
    func: Callable,
    arg_ordering: list[sym.Symbol],
    init_unknowns: dict[sym.Symbol, float],
    abs_tolerance: float | None,
    bounds: dict[sym.Symbol, tuple[float, float]] | None,
    **kwargs,
) -> tuple[dict[sym.Symbol, float], optimize.OptimizeResult]:
    """See `solve_mixing`.

    This function also returns the raw scipy.optimize.OptimizeResult for
    debugging help.

    :param func: The vector function
    :type func: Callable,
    :param arg_ordering: Call order defined by the Sympy lambdification
    :type arg_ordering: list[sympy.Symbol]
    :param init_unknowns: The unknowns and their initial guesses for the solver
    :type init_unknowns: dict[sympy.Symbol, float]
    :param abs_tolerance: Absolute tolerance of the result from zero (disable check with None)
    :type abs_tolerance: float | None
    :param bounds: Bounds for each of the unknown arguments
    :type bounds: dict[sympy.Symbol, tuple[float, float]] | None
    :return: Dictionary of solved values and the raw OptimizeResult output
    :rtype: tuple[dict[sympy.Symbol, float], scipy.optimize.OptimizeResult]
    """
    unknown_args = [k for k in arg_ordering if k in init_unknowns]

    bounds = _DEFAULT_BOUNDS if bounds is None else bounds
    kwargs["bounds"] = [bounds[k][0] for k in unknown_args], [
        bounds[k][1] for k in unknown_args
    ]
    # default tolerances for least_squares if they weren't set by user
    for k in ["ftol", "gtol", "xtol"]:
        if k not in kwargs:
            kwargs[k] = 1e-15
    solve = _least_sqr_tolerance(
        func, arg_ordering, init_unknowns, abs_tolerance, **kwargs
    )
    return solve


# %%
def solve_mixing(
    func: Callable,
    arg_ordering: list[sym.Symbol],
    init_unknowns: dict[sym.Symbol, float],
    abs_tolerance: float | None = 1e-8,
    bounds: dict[sym.Symbol, tuple[float, float]] | None = None,
    **kwargs,
) -> dict[sym.Symbol, float]:
    """Solves the provided vector function for unknowns in the mixing equation.

    The absolute tolerance is used to check the accuracy of the solution, which
    will raise a ValueError if any equation in the system is too far from 0.
    More options can be passed along to `scipy.optimize.least_squares` using
    keyword arguments.

    :param func: The vector function
    :type func: Callable
    :param arg_ordering: Call order defined by the Sympy lambdification
    :type arg_ordering: list[sympy.Symbol]
    :param init_unknowns: The unknowns and their initial guesses for the solver
    :type init_unknowns: dict[sympy.Symbol, float]
    :param abs_tolerance: Absolute tolerance of the result from zero (disable check with None), defaults to 1e-8
    :type abs_tolerance: float | None, optional
    :param bounds: Bounds for each of the unknown arguments, defaults to None
    :type bounds: dict[sympy.Symbol, tuple[float, float]] | None, optional
    :return: Dictionary of solved values
    :rtype: dict[sympy.Symbol, float]
    """
    return debug_solve_mixing(
        func, arg_ordering, init_unknowns, abs_tolerance, bounds, **kwargs
    )[0]


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
        ALL_SYMBOL["prop"]["A"]: m_axial,
        ALL_SYMBOL["bare"]["pi0-A"]: m_pi0_axial,
        ALL_SYMBOL["bare"]["eta-A"]: m_eta_axial,
        ALL_SYMBOL["bare"]["eta0-A"]: m_eta0_axial,
    }
    uvw = []
    for s in kinetic_sines():
        call = list(s.free_symbols)
        uvw.append(sym.lambdify(call, sym.asin(s))(*[parameter[k] for k in call]))
    return tuple(uvw)


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
        ALL_SYMBOL["coupling"]["phi-chi"]: bare_mediator_dm_coupling,
        ALL_SYMBOL["angle"]["alpha"]: alpha,
        ALL_SYMBOL["angle"]["beta"]: beta,
        ALL_SYMBOL["angle"]["gamma"]: gamma,
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
        ALL_SYMBOL["rotated"]["pi-gamma"]: known_pion_photon_coupling,
        ALL_SYMBOL["rotated"]["eta-gamma"]: known_eta_photon_coupling,
        ALL_SYMBOL["angle"]["alpha"]: alpha,
        ALL_SYMBOL["angle"]["beta"]: beta,
        ALL_SYMBOL["angle"]["gamma"]: gamma,
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
        ALL_SYMBOL["parameters"]["Epi"]: energy_pion,
        ALL_SYMBOL["prop"]["pi0"]: mass_pion,
        ALL_SYMBOL["prop"]["K0"]: mass_kaon,
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
        ALL_SYMBOL["coupling"]["pi-pi-K"]: bare_pion_pion_kaon_coupling,
        ALL_SYMBOL["coupling"]["phi-pi-K"]: bare_mediator_pion_kaon_coupling,
        ALL_SYMBOL["coupling"]["eta-pi-K"]: bare_eta_pion_kaon_coupling,
        ALL_SYMBOL["angle"]["alpha"]: alpha,
        ALL_SYMBOL["angle"]["beta"]: beta,
        ALL_SYMBOL["angle"]["gamma"]: gamma,
    }
    coupling = []
    for g in rotated_pseudo_kaon_couplings():
        call = list(g.free_symbols)
        coupling.append(sym.lambdify(call, g)(*[parameter[k] for k in call]))
    return tuple(coupling)
