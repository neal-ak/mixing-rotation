"""Contains the numerical solver for the mixing equation."""

# %%
from collections.abc import Callable

# %%
import numpy as np
from scipy import optimize
import sympy as sym

# %%
from .symbolic import PROP, BARE, ANGLE

# %%
_DEFAULT_BOUNDS = {v: [0, np.inf] for v in list(PROP.values()) + list(BARE.values())} | {v: [-np.pi/2, np.pi/2] for v in ANGLE.values()}

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
        **{k:v for k,v in kwargs.items() if k not in ["fun", "x0"]},
    )
    solve = dict(zip(unknown_args, solution.x))

    if abs_tolerance is not None:
        correct = np.isclose(
            function([solve[k] for k in arg_ordering if k in solve]),
            [0] * len(solution.x),
            atol=abs_tolerance,
        )
        if not np.all(correct):
            raise ValueError("Solutions not close enough to zero")
    return solve, solution


# %%
def initialize_system(equation: sym.Matrix) -> tuple[list[sym.Symbol], list[Callable], list[dict[sym.Symbol,Callable]]]:
    """Initializes a system of equations for the solver.

    Sets up the argument call order for the lambdified sympy equations, the
    equations themselves and their partials. Due to redundancy the callable
    system drops the lower diagonal component of the matrix (6 equations in the
    system).

    :param equation: Matrix representing the system of equations
    :type equation: sym.Matrix
    :return: Argument call order, system of equations, and all partials for each equation
    :rtype: tuple[list[sympy.Symbol], list[Callable], list[dict[sympy.Symbol,Callable]]]
    """
    arg_ordering = [v for v in list(PROP.values())+list(BARE.values())+list(ANGLE.values()) if v in equation.free_symbols]
    system, partials = [], []
    for eqn in [
        e for i, e in enumerate(equation) if i not in [3, 6, 7]
    ]:
        system.append(sym.lambdify(arg_ordering, eqn))
        partials.append({arg: sym.lambdify(arg_ordering, sym.diff(eqn, arg)) for arg in arg_ordering})
    return arg_ordering, system, partials


# %%
def create_func_jac(
    system: list[Callable],
    arg_ordering: list[sym.Symbol],
    arg_unknown: list[sym.Symbol],
    knowns: dict[sym.Symbol, float],
    partials: list[dict[sym.Symbol, Callable]] = None,
) -> tuple[Callable, Callable | None]:
    """Creates the vector function and jacobian for the Scipy solver.

    :param system: System of equations lambdified from Sympy
    :type system: list[Callable]
    :param arg_ordering: Call order defined by the Sympy lambdification
    :type arg_ordering: list[sympy.Symbol]
    :param arg_unknown: The unknowns as a subset of the call order
    :type arg_unknown: list[sympy.Symbol]
    :param knowns: Known arguments and their values in the system
    :type knowns: dict[sympy.Symbol, float]
    :param partials: Partials of the system sorted the same as the system of equations, defaults to None
    :type partials: list[dict[sympy.Symbol, Callable]], optional
    :return: The function and jacobian Scipy can use to optimize
    :rtype: tuple[Callable, Callable | None]
    """
    def function(x):
        calling = _build_call(arg_ordering, x, knowns)
        return [f(*calling) for f in system]

    if partials is not None:
        def jacobian(x):
            calling = _build_call(arg_ordering, x, knowns)
            return [[part[k](*calling) for k in arg_unknown] for part in partials]
        return function, jacobian
    return function, None


# %%
def debug_solve_mixing(
    func_jac: tuple[Callable,Callable|None],
    arg_ordering: list[sym.Symbol],
    init_unknowns: dict[sym.Symbol, float],
    abs_tolerance: float | None,
    bounds: dict[sym.Symbol, tuple[float, float]] | None,
    **kwargs
) -> tuple[dict[sym.Symbol, float], optimize.OptimizeResult]:
    """See `solve_mixing`.
    
    This function also returns the raw scipy.optimize.OptimizeResult for
    debugging help.

    :param func_jac: The vector function and its jacobian
    :type func_jac: tuple[Callable,Callable | None]
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

    if func_jac[1] is not None:
        kwargs["jac"] = func_jac[1]
    bounds = _DEFAULT_BOUNDS if bounds is None else bounds
    kwargs["bounds"] = [bounds[k][0] for k in unknown_args], [bounds[k][1] for k in unknown_args]
    # default tolerances for least_squares if they weren't set by user
    for k in ["ftol", "gtol", "xtol"]:
        if k not in kwargs:
            kwargs[k] = 1e-15
    solve = _least_sqr_tolerance(func_jac[0], arg_ordering, init_unknowns, abs_tolerance, **kwargs)
    return solve


# %%
def solve_mixing(
    func_jac: tuple[Callable,Callable|None],
    arg_ordering: list[sym.Symbol],
    init_unknowns: dict[sym.Symbol, float],
    abs_tolerance: float | None = 1e-8,
    bounds: dict[sym.Symbol, tuple[float, float]] | None = None,
    **kwargs
) -> dict[sym.Symbol, float]:
    """Solves the provided vector function for unknowns in the mixing equation.
    
    The absolute tolerance is used to check the accuracy of the solution, which
    will raise a ValueError if any equation in the system is too far from 0.
    More options can be passed along to `scipy.optimize.least_squares` using
    keyword arguments.

    :param func_jac: The vector function and its jacobian
    :type func_jac: tuple[Callable,Callable | None]
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
    return debug_solve_mixing(func_jac, arg_ordering, init_unknowns, abs_tolerance, bounds, **kwargs)[0]
