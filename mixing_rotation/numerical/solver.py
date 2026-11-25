"""Contains the numerical solver for the mixing equation."""

# %%
from collections.abc import Callable
import warnings

# %%
import numpy as np
from scipy import optimize
import sympy as sym

# %%
from ..symbolic.general import ANGLE, MASS, COUPLING, PARAMETERS

# %%
#: The pion decay constant from `Holstein:1990ua <https://inspirehep.net/literature/303682>`_.
PION_DECAY = 0.0924
#: The quark masses from `PDG <https://pdglive.lbl.gov/Particle.action?node=Q123>`_.
MASS_QUARK = {"up": 2.16e-3, "down": 4.7e-3, "strange": 93.5e-3}
#: The neutral pion mass from `PDG <https://pdglive.lbl.gov/Particle.action?node=S009>`_.
MASS_NEUTRAL_PION = 0.1349768
#: The charged pion mass from `PDG <https://pdglive.lbl.gov/Particle.action?node=S008>`_.
MASS_CHARGED_PION = 0.13957039
#: The charged kaon mass from `PDG <https://pdglive.lbl.gov/Particle.action?node=S010>`_.
MASS_CHARGED_KAON = 0.493677
#: The eta mass from `PDG <https://pdglive.lbl.gov/Particle.action?node=S014>`_.
MASS_ETA = 0.547862
#: The eta prime mass from `PDG <https://pdglive.lbl.gov/Particle.action?node=M002>`_.
MASS_ETA_PRIME = 0.95778
#: The branching ratio of eta prime decay to two photons from
#: `PDG <https://pdglive.lbl.gov/BranchingRatio.action?desig=6&parCode=M002>`_.
DECAY_ETA_PRIME_2GAMMA = 4.34e-06
#: Eq. (`27 <https://arxiv.org/pdf/2611.xxxxx#equation.2.27>`_) in [`arXiv:2611.xxxxx <https://arxiv.org/abs/2611.xxxxx>`_].
ETA_PRIME_DECAY = (
    7.2973525693e-3 * MASS_ETA_PRIME / 8 / np.sqrt(np.pi) ** 3 / DECAY_ETA_PRIME_2GAMMA
)
#: See :py:const:`mixing_rotation.numerical.solver.ETA_PRIME_DECAY`.
G_PI_GAMMA = 7.2973525693e-3 / (np.pi * PION_DECAY)
#: See :py:const:`mixing_rotation.numerical.solver.ETA_PRIME_DECAY`.
G_ETA_GAMMA = G_PI_GAMMA / np.sqrt(3)
#: See :py:const:`mixing_rotation.numerical.solver.ETA_PRIME_DECAY`.
G_ETA_PRIME_GAMMA = G_PI_GAMMA * PION_DECAY / ETA_PRIME_DECAY
#: Below Eq. (`3 <https://arxiv.org/pdf/2611.xxxxx#equation.2.3>`_) in
#: [`arXiv:2611.xxxxx <https://arxiv.org/abs/2611.xxxxx>`_].
B_0 = MASS_CHARGED_PION**2 / (MASS_QUARK["up"] + MASS_QUARK["down"])
#: Eq. (`5 <https://arxiv.org/pdf/2611.xxxxx#equation.2.5>`_) in
#: [`arXiv:2611.xxxxx <https://arxiv.org/abs/2611.xxxxx>`_].
L_4 = 1.87e-3
#: See :py:const:`mixing_rotation.numerical.solver.L_4`.
L_5 = 1.22e-3
#: See :py:const:`mixing_rotation.numerical.solver.L_4`.
L_6 = 1.46e-3
#: See :py:const:`mixing_rotation.numerical.solver.L_4`.
L_7 = -0.39e-3
#: Below Eq. (`8 <https://arxiv.org/pdf/2611.xxxxx#equation.2.8>`_) in
#: [`arXiv:2611.xxxxx <https://arxiv.org/abs/2611.xxxxx>`_].
G_8 = 3.62
#: See :py:const:`mixing_rotation.numerical.solver.G_8`.
G_8S = -0.38 * G_8


# %%
class PhysicsWarning(Warning):
    """Warning that the physical veracity is in doubt."""


# %%
def _build_call(
    arg_ordering: list[sym.Symbol],
    x0: list[float],
    knowns: dict[sym.Symbol, float],
) -> list[float]:
    """Create call signature for lambdified Sympy functions.

    :param arg_ordering: Call order defined by the Sympy lambdification.
    :type arg_ordering: list[sym.Symbol]
    :param x0: Values for unknown parameters.
    :type x0: list[float]
    :param knowns: Known arguments and their fixed values.
    :type knowns: dict[sym.Symbol, float]
    :return: Call signature.
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
    **kwargs: dict,
) -> tuple[dict[sym.Symbol, float], optimize.OptimizeResult]:
    """Runs least squares optimization and checks solution for validity.

    The tolerance check can be disabled by setting the tolerance to None.

    :param function: Function that least squares will optimize.
    :type function: Callable
    :param arg_ordering: Call order defined by the Sympy lambdification.
    :type arg_ordering: list[sym.Symbol]
    :param init_unknowns: Unknown arguments and their initial guesses.
    :type init_unknowns: dict[sym.Symbol, float]
    :param abs_tolerance: Absolute tolerance for the check, disable with None.
    :type abs_tolerance: float | None
    :param kwargs: Keyword arguments passed along to :py:func:`scipy.optimize.least_squares`.
    :type kwargs: dict
    :raises ValueError: Raised if solution outside tolerance.
    :return: Dictionary of solved values and the raw OptimizeResult output.
    :rtype: tuple[dict[sym.Symbol, float], optimize.OptimizeResult]
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
    system drops the lower diagonal component of the matrix (it is a symmetric
    matrix).

    :param equation: Matrix representing the system of equations.
    :type equation: sym.Matrix
    :return: Argument call order, and the system of equations.
    :rtype: tuple[list[sym.Symbol], list[Callable]]
    :raises ValueError: Raised when the equation is not symmetric.
    """
    if not equation.is_symmetric():
        raise ValueError("The equation matrix must be symmetric")
    arg_ordering = []
    for d in [ANGLE, MASS, COUPLING, PARAMETERS]:
        arg_ordering += [v for v in d.values() if v in equation.free_symbols]
    system = []
    for r in range(equation.rows):
        for c in range(r, equation.cols):
            system.append(sym.lambdify(arg_ordering, equation[r, c]))
    return arg_ordering, system


# %%
def create_func(
    system: list[Callable],
    arg_ordering: list[sym.Symbol],
    knowns: dict[sym.Symbol, float],
) -> Callable:
    """Creates the vector function for the Scipy solver.

    :param system: System of equations lambdified from Sympy.
    :type system: list[Callable]
    :param arg_ordering: Call order defined by the Sympy lambdification.
    :type arg_ordering: list[sym.Symbol]
    :param knowns: Known arguments and their values in the system.
    :type knowns: dict[sym.Symbol, float]
    :raises ValueError: Raised if bare pseudo scalar mass input less than zero.
    :return: The function Scipy can use to optimize.
    :rtype: Callable
    """
    if MASS["phiBare"] in knowns and knowns[MASS["phiBare"]] < 0:
        raise ValueError(
            "Pseudo scalar mediator mass should be non-negative"
            + f" ({knowns[MASS['phiBare']]:.4e} GeV)"
        )

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
    **kwargs: dict,
) -> tuple[dict[sym.Symbol, float], optimize.OptimizeResult]:
    """See `solve_mixing`.

    This function also returns the raw :py:class:`scipy.optimize.OptimizeResult` for
    debugging help.

    :param func: The vector function.
    :type func: Callable
    :param arg_ordering: Call order defined by the Sympy lambdification.
    :type arg_ordering: list[sym.Symbol]
    :param init_unknowns: The unknowns and their initial guesses for the solver.
    :type init_unknowns: dict[sym.Symbol, float]
    :param abs_tolerance: Absolute tolerance of the result from zero (disable check with None).
    :type abs_tolerance: float | None
    :param bounds: Bounds for each of the unknown arguments.
    :type bounds: dict[sym.Symbol, tuple[float, float]] | None
    :param kwargs: Keyword arguments passed to `_least_sqr_tolerance`.
    :type kwargs: dict
    :raises PhysicsWarning: Raised if solution values outside physically permitted ranges.
    :return: Dictionary of solved values and the raw OptimizeResult output.
    :rtype: tuple[dict[sym.Symbol, float], optimize.OptimizeResult]
    """
    unknown_args = [k for k in arg_ordering if k in init_unknowns]

    if bounds is None:
        kwargs["bounds"] = [], []
        for arg in unknown_args:
            if arg in MASS.values():
                kwargs["bounds"][0].append(0)
                kwargs["bounds"][1].append(np.inf)
            elif arg in ANGLE.values():
                kwargs["bounds"][0].append(-np.pi)
                kwargs["bounds"][1].append(np.pi)
            else:
                kwargs["bounds"][0].append(-np.inf)
                kwargs["bounds"][1].append(np.inf)
    else:
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
    if MASS["phi"] in solve[0] and solve[0][MASS["phi"]] < 0:
        warnings.warn(
            "This combination of model parameters produces a negative pseudo scalar observed"
            + f" (propagating) mass ({solve[0][MASS['phi']]:.4e} GeV). The coupling parameters"
            + " leading to mixing are too large compared to the difference between the mass of"
            + " the mediator and the mass of the mesons it mixes with. The model is unstable with"
            + " these parameters and predictions for the effective coupling in the rotated basis"
            + " should not be trusted!",
            PhysicsWarning,
        )
    for meson, mass in {
        "pi0": MASS_NEUTRAL_PION,
        "eta": MASS_ETA,
        "etaP": MASS_ETA_PRIME,
    }.items():
        if MASS[meson + "Bare"] in solve[0]:
            if solve[0][MASS[meson + "Bare"]] < 0:
                warnings.warn(
                    f"This combination of model parameters produces a negative bare {meson} mass"
                    + f" ({solve[0][MASS[meson + 'Bare']]:.4e} GeV). The coupling parameters"
                    + " leading to mixing are too large compared to the difference between the"
                    + " mass of the mediator and the mass of the mesons it mixes with. The model"
                    + " is unstable with these parameters and predictions for the effective"
                    + " coupling in the rotated basis should not be trusted!",
                    PhysicsWarning,
                )
            max_rotation = MASS_CHARGED_PION - MASS_NEUTRAL_PION
            if np.abs(solve[0][MASS[meson + "Bare"]] - mass) > max_rotation:
                warnings.warn(
                    f"This combination of model parameters predicts a bare {meson} mass"
                    + f" ({solve[0][MASS[meson + 'Bare']]:.4e} GeV) that differs by more than"
                    + f" {max_rotation:.4e} GeV from its observed value. This may be in conflict"
                    + " with some predictions from the ChEFT Lagrangian!",
                    PhysicsWarning,
                )
    return solve


# %%
def solve_mixing(
    func: Callable,
    arg_ordering: list[sym.Symbol],
    init_unknowns: dict[sym.Symbol, float],
    abs_tolerance: float | None = 1e-8,
    bounds: dict[sym.Symbol, tuple[float, float]] | None = None,
    **kwargs: dict,
) -> dict[sym.Symbol, float]:
    """Solves the provided vector function for unknowns in the mixing equation.

    The absolute tolerance is used to check the accuracy of the solution, which
    will raise a ValueError if any equation in the system is too far from 0.
    More options can be passed along to :py:func:`scipy.optimize.least_squares` using
    keyword arguments.

    :param func: The vector function.
    :type func: Callable
    :param arg_ordering: Call order defined by the Sympy lambdification.
    :type arg_ordering: list[sym.Symbol]
    :param init_unknowns: The unknowns and their initial guesses for the solver.
    :type init_unknowns: dict[sym.Symbol, float]
    :param abs_tolerance: Absolute tolerance of the result from zero (disable check with None), \
        defaults to 1e-8.
    :type abs_tolerance: float | None, optional
    :param bounds: Bounds for each of the unknown arguments, defaults to None.
    :type bounds: dict[sym.Symbol, tuple[float, float]] | None, optional
    :param kwargs: Keyword arguments passed to "py:func:`_least_sqr_tolerance`.
    :type kwargs: dict
    :return: Dictionary of solved values.
    :rtype: dict[sym.Symbol, float]
    """
    return debug_solve_mixing(
        func, arg_ordering, init_unknowns, abs_tolerance, bounds, **kwargs
    )[0]
