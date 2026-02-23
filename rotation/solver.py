"""Contains the numerical solver for the mixing equation."""

# %%
import numpy as np
from scipy import optimize
import sympy as sym

# %%
from .symbolic import MASS, MIX, ANGLE, create_mix_equation, create_3d_rotation

# %%
def optimize_system(
    system: list,
    arg_order: list[sym.Symbol],
    unknowns: list[float],
    knowns: dict[sym.Symbol, float],
):
    """
    Evaluate the system of equations with a fixed argument ordering. The
    unknowns are a list of numbers passed to the functions in the system. Their
    ordering matches the argument ordering. The knowns are pairs of Sympy
    Symbols and their fixed values. There should be 6 unknown parameters, and
    as a result there should be len(arg_order)-6 knowns.
    """
    if len(unknowns) != 6:
        raise ValueError("The system can only accommodate 6 unknowns")

    # Create a call signature to the lambdified Sympy functions:
    # A. Pull from the knowns
    # B. If that fails, get the next unknown that was passed in
    unknowns = iter(unknowns)
    call = {}
    for argument in arg_order:
        if argument in knowns:
            call[argument] = knowns[argument]
        else:
            call[argument] = next(unknowns)

    arguments = [call[argument] for argument in arg_order]
    return [f(*arguments) for f in system]


# %%
def solve_mixing(
    system: list,
    arguments: list[sym.Symbol],
    unknowns: dict[sym.Symbol, float],
    knowns: dict[sym.Symbol, float],
    tolerance: float | None = None,
    bounds: None|dict[sym.Symbol, tuple[float, float]] = None,
    info: bool = False,
    **kwargs
) -> dict[sym.Symbol, float]:
    """
    Performs the Scipy optimization routine, and packages the output into a
    dictionary of Symbols (from the input unknowns) mapped to their solved
    values, also optionally returning the raw OptimizeResult from the
    least squares with extra information. Also can perform a check to make sure
    that the solution is close enough (within float tolerances) of zero when
    subbed back into the system of equations. If it fails this check, it will
    raise a ValueError. The absolute tolerance of this check can be configured.
    The bounds can be defined by passing a dictionary of the Symbols with their
    lower and upper bound pairs. Keyword arguments (except for 'fun', 'x0', and
    'jac') are passed along to scipy.optimize.least_squares.
    """

    def func(x):
        return optimize_system(system, arguments, x, knowns)

    unknown_args = [k for k in arguments if k in unknowns]
    if bounds is not None:
        kwargs["bounds"] = [bounds[k][0] for k in unknown_args], [bounds[k][1] for k in unknown_args]
    solution = optimize.least_squares(
        func,
        [unknowns[k] for k in unknown_args],
        # jac = ,
        **{k:v for k,v in kwargs.items() if k not in ["fun", "x0", "jac"]}
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
def initialize() -> tuple[sym.Matrix, list[sym.Symbol], list]:
    """
    Initializes the system of equations, and the argument order to the
    equations in the system. Returns the argument order, the system of
    equations, and the matrix equation as a tuple.
    """
    equation, _ = create_mix_equation(create_3d_rotation()[0])
    arguments = list(MASS.values()) + list(MIX.values()) + list(ANGLE.values())
    system = []
    for eq in [
        eq for i, eq in enumerate(equation.lhs - equation.rhs) if i not in [3, 6, 7]
    ]:
        system.append(sym.lambdify(arguments, eq))
    return arguments, system, equation
