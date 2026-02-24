"""Contains the numerical solver for the mixing equation."""

# %%
import numpy as np
from scipy import optimize
import sympy as sym

# %%
from .symbolic import MASS, MIX, ANGLE, create_mix_equation, create_3d_rotation

# %%
def build_call(
    arg_order: list[sym.Symbol],
    unknowns: list[float],
    knowns: dict[sym.Symbol, float],
):
    """
    Takes the arguments split between unknowns and knowns and creates a
    complete call signature for the lambdified Sympy functions. There should
    be 6 unknown parameters, and as a result there should be len(arg_order)-6
    knowns.
    """
    if len(unknowns) != 6:
        raise ValueError("The system can only accommodate 6 unknowns")

    unknowns = iter(unknowns)
    call = {}
    # A. Pull from the knowns
    # B. If that fails, get the next unknown that was passed in
    for argument in arg_order:
        if argument in knowns:
            call[argument] = knowns[argument]
        else:
            call[argument] = next(unknowns)

    return [call[argument] for argument in arg_order]

# %%
def solve_mixing(
    system: list,
    arguments: list[sym.Symbol],
    unknowns: dict[sym.Symbol, float],
    knowns: dict[sym.Symbol, float],
    partials: list[dict[sym.Symbol, any]] = None,
    tolerance: float | None = 1e-8,
    bounds: None | dict[sym.Symbol, tuple[float, float]] = None,
    info: bool = False,
    **kwargs
) -> dict[sym.Symbol, float]:
    """
    Performs the Scipy optimization routine, and packages the output into a
    dictionary of Symbols (from the input unknowns) mapped to their solved
    values, also optionally returning the raw OptimizeResult from the
    least squares with extra information. Optionally can use an analytic
    Jacobian by passing the list of all partial derivatives in a list of
    dictionaries sorted in the same order as the system's equation list. If
    this is not done, least_squares makes a numerical approximation. Also can
    perform a check to make sure that the solution is close enough (within
    float tolerances) of zero when subbed back into the system of equations.
    If it fails this check, it will raise a ValueError. The absolute tolerance
    of this check can be configured. The bounds can be defined by passing a
    dictionary of the Symbols with their lower and upper bound pairs. Keyword
    arguments (except for 'fun' and 'x0') are passed along to
    scipy.optimize.least_squares.
    """

    def func(x):
        calling = build_call(arguments, x, knowns)
        return [f(*calling) for f in system]

    unknown_args = [k for k in arguments if k in unknowns]
    if partials is not None:
        def jacobian(x):
            calling = build_call(arguments, x, knowns)
            return [[part[k](*calling) for k in unknown_args] for part in partials]
        kwargs["jac"] = jacobian

    if bounds is None:
        bounds = {
            MASS["pi"]: [0, np.inf],
            MASS["eta"]: [0, np.inf],
            MASS["phi"]: [0, np.inf],
            MIX["pi"]: [0, np.inf],
            MIX["eta"]: [0, np.inf],
            MIX["phi"]: [0, np.inf],
            MIX["pi-phi"]: [0, np.inf],
            MIX["eta-phi"]: [0, np.inf],
            ANGLE["alpha"]: [-np.pi/2, np.pi/2],
            ANGLE["gamma"]: [-np.pi/2, np.pi/2],
            ANGLE["beta"]: [-np.pi/2, np.pi/2],
        }
    kwargs["bounds"] = [bounds[k][0] for k in unknown_args], [bounds[k][1] for k in unknown_args]
    # default tolerances for least_squares if they weren't set by user
    for k in ["ftol", "gtol", "xtol"]:
        if k not in kwargs:
            kwargs[k] = 1e-15
    solution = optimize.least_squares(
        func,
        [unknowns[k] for k in unknown_args],
        **{k:v for k,v in kwargs.items() if k not in ["fun", "x0"]}
    )
    solve = dict(zip(unknown_args, solution.x))
    if tolerance is not None:
        correct = np.isclose(
            func([solve[k] for k in arguments if k in solve]), [0] * len(solution.x),
            atol=tolerance,
        )
        if not np.all(correct):
            raise ValueError(f"Solutions not close enough to zero: {correct}")
    if info:
        return solve, solution
    return solve


# %%
def initialize() -> tuple[list[sym.Symbol], list, list, sym.Matrix]:
    """
    Initializes the system of equations, and the argument order to the
    equations in the system. Returns the argument order, the system of
    equations, all the partial derivatives and the matrix equation as a tuple.
    """
    equation, _ = create_mix_equation(create_3d_rotation()[0])
    arguments = list(MASS.values()) + list(MIX.values()) + list(ANGLE.values())
    system,partials = [], []
    for eq in [
        eq for i, eq in enumerate(equation.lhs - equation.rhs) if i not in [3, 6, 7]
    ]:
        system.append(sym.lambdify(arguments, eq))
        partials.append({arg: sym.lambdify(arguments, sym.diff(eq, arg)) for arg in arguments})
    return arguments, system, partials, equation
