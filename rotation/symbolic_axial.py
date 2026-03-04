"""Symbolically computes the axial mixing equation."""

# %%
import sympy as sym

# %%
from .symbolic import ALL_SYMBOL


# %%
def create_kinetic() -> sym.Matrix:
    """Creates the kinetic mixing matrix for the axial vector mediator.

    :return: Kinetic mixing matrix
    :rtype: sym.Matrix
    """
    angles = [ALL_SYMBOL["angle"][k] for k in ["u", "v", "w"]]
    d = sym.sqrt(
        1
        - sym.sin(angles[0]) ** 2
        - sym.sin(angles[1]) ** 2
        - sym.sin(angles[2]) ** 2
        + 2 * sym.sin(angles[0]) * sym.sin(angles[1]) * sym.sin(angles[2])
    )
    return sym.Matrix(
        [
            [
                1,
                -sym.tan(angles[0]),
                (
                    sym.tan(angles[0]) * sym.sin(angles[2])
                    - sym.sin(angles[1]) / sym.cos(angles[0])
                )
                / d,
            ],
            [
                0,
                1 / sym.cos(angles[0]),
                (
                    sym.tan(angles[0]) * sym.sin(angles[1])
                    - sym.sin(angles[2]) / sym.cos(angles[0])
                )
                / d,
            ],
            [0, 0, sym.cos(angles[0]) / d],
        ]
    )


# %%
def create_axial_mix_equation(
    rotation: sym.Matrix, kinetic: sym.Matrix
) -> tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix]]:
    """Creates the axial vector mixing matrix equation.

    :param rotation: Rotation matrix
    :type rotation: sym.Matrix
    :param kinetic: Kinetic rotation matrix
    :type kinetic: sym.Matrix
    :return: Mixing matrix equation, (kinetic transpose, rotation transpose, bare mass matrix, propagation mass matrix)
    :rtype: tuple[sym.Matrix, tuple[sym.Matrix, sym.Matrix, sym.Matrix, sym.Matrix]]
    """
    rotation_transpose = rotation.transpose()
    kinetic_transpose = kinetic.transpose()
    bare_matrix = sym.Matrix(
        [
            [ALL_SYMBOL["bare"]["pi0"] ** 2, 0, 0],
            [0, ALL_SYMBOL["bare"]["eta"] ** 2, 0],
            [0, 0, ALL_SYMBOL["bare"]["eta0"] ** 2],
        ]
    )
    prop_matrix = sym.Matrix(
        [
            [ALL_SYMBOL["prop"]["pi0"] ** 2, 0, 0],
            [0, ALL_SYMBOL["prop"]["eta"] ** 2, 0],
            [0, 0, ALL_SYMBOL["prop"]["eta0"] ** 2],
        ]
    )
    mixing_equation = sym.Eq(
        prop_matrix,
        sym.MatMul(
            rotation, kinetic_transpose, bare_matrix, kinetic, rotation_transpose
        ),
        evaluate=False,
    )
    return mixing_equation, (
        kinetic_transpose,
        rotation_transpose,
        bare_matrix,
        prop_matrix,
    )


# %%
def _kinetic_sine(m1, m2) -> sym.core.mul.Mul:
    """Helper to automate sine of kinetic mixing angle."""
    ma = ALL_SYMBOL["prop"]["A"]
    return (
        -m1 * m2 / (2 * ma**2 * sym.sqrt((1 - (m1 / ma) ** 2) * (1 - (m2 / ma) ** 2)))
    )


# %%
def kinetic_sines() -> tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]:
    """Creates the sines of the three kinetic mixing angles.

    :return: sine(u), sine(v), sine(w)
    :rtype: tuple[sym.core.mul.Mul, sym.core.mul.Mul, sym.core.mul.Mul]
    """
    s_u = _kinetic_sine(ALL_SYMBOL["bare"]["pi0-A"], ALL_SYMBOL["bare"]["eta-A"])
    s_v = _kinetic_sine(ALL_SYMBOL["bare"]["pi0-A"], ALL_SYMBOL["bare"]["eta0-A"])
    s_w = _kinetic_sine(ALL_SYMBOL["bare"]["eta0-A"], ALL_SYMBOL["bare"]["eta-A"])
    return s_u, s_v, s_w
