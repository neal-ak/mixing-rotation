# %%
import pathlib
import pickle
import warnings

import numpy as np
import sympy as sym

from rotation.numerical import solver, axial
from rotation.symbolic import general
from rotation.symbolic.axial import create_axial_mix_equation, create_kinetic


# %%
def small_alpha(m_AP, m_pi0_AP):
    return 0.5 * m_pi0_AP**2 / (solver.MASS_NEUTRAL_PION**2 - m_AP**2)


def small_beta(m_AP, m_eta_AP):
    return 0.5 * m_eta_AP**2 / (solver.MASS_ETA**2 - m_AP**2)


def small_gamma(m_AP, m_etaP_AP):
    return 0.5 * m_etaP_AP**2 / (solver.MASS_ETA_PRIME**2 - m_AP**2)


def scan_quarks(
    coupling_strength: list[float],
    tolerance: float,
    quarks: tuple[float, float, float],
    m_axial_bare: float,
    old_results: dict[str | sym.Symbol, np.NDArray] = None,
) -> dict[str | sym.Symbol, np.NDArray]:
    new_results = {"coupling": np.array(coupling_strength)}
    proximity_thresh = 1 / 3

    if old_results is not None:  # We're gonna remove duplicates
        matches = np.abs(
            old_results["coupling"][np.newaxis, ...]
            / new_results["coupling"][..., np.newaxis]
            - 1
        )
        new_results["coupling"] = new_results["coupling"][
            ~(np.any(matches == 0, axis=1))
        ]
        if len(new_results["coupling"]) == 0:
            return old_results
        sort_args = np.argsort(matches, axis=-1)
        sort_args = np.ma.array(
            sort_args,
            mask=np.take_along_axis(matches, sort_args, axis=-1) >= proximity_thresh,
        )
    equation = create_axial_mix_equation(
        general.create_3d_rotation()[0], create_kinetic()
    )[0]
    argument_order, system_of_equations = solver.initialize_system(
        equation.lhs - equation.rhs
    )

    new_results |= {
        k: np.nan * np.empty(new_results["coupling"].shape) for k in argument_order
    }

    quarks = np.array(quarks[:3])
    diagonals = axial.axial_calculate_bare_masses(
        *(new_results["coupling"] * quarks[:, np.newaxis])
    )
    kin_u, kin_v, kin_w = axial.calculate_uvw(
        m_axial_bare,
        diagonals[general.MASS["pi0-AP"]],
        diagonals[general.MASS["eta-AP"]],
        diagonals[general.MASS["etaP-AP"]],
    )
    known = {
        general.MASS["pi0"]: solver.MASS_NEUTRAL_PION,
        general.MASS["eta"]: solver.MASS_ETA,
        general.MASS["etaP"]: solver.MASS_ETA_PRIME,
        general.ANGLE["u"]: kin_u[0],
        general.ANGLE["v"]: kin_v[0],
        general.ANGLE["w"]: kin_w[0],
    }
    unknown = {
        general.MASS["pi0Bare"]: solver.MASS_NEUTRAL_PION,
        general.MASS["etaBare"]: solver.MASS_ETA,
        general.MASS["etaPBare"]: solver.MASS_ETA_PRIME,
        general.ANGLE["alpha"]: (
            0.0
            if not np.isfinite(
                small_alpha(m_axial_bare, diagonals[general.MASS["pi0-AP"]][0])
            )
            else small_alpha(m_axial_bare, diagonals[general.MASS["pi0-AP"]][0])
        ),
        general.ANGLE["beta"]: (
            0.0
            if not np.isfinite(
                small_beta(m_axial_bare, diagonals[general.MASS["eta-AP"]][0])
            )
            else small_beta(m_axial_bare, diagonals[general.MASS["eta-AP"]][0])
        ),
        general.ANGLE["gamma"]: (
            0.0
            if not np.isfinite(
                small_gamma(m_axial_bare, diagonals[general.MASS["etaP-AP"]][0])
            )
            else small_gamma(m_axial_bare, diagonals[general.MASS["etaP-AP"]][0])
        ),
    }

    num_fails, previous, successes = None, None, []
    for index, x in enumerate(new_results["coupling"]):
        if num_fails is not None and num_fails > 1:
            break
        known |= {
            general.ANGLE["u"]: kin_u[index],
            general.ANGLE["v"]: kin_v[index],
            general.ANGLE["w"]: kin_w[index],
        }
        unknown_update = {
            general.ANGLE["alpha"]: (
                0.0
                if not np.isfinite(
                    small_alpha(m_axial_bare, diagonals[general.MASS["pi0-AP"]][index])
                )
                else small_alpha(m_axial_bare, diagonals[general.MASS["pi0-AP"]][index])
            ),
            general.ANGLE["beta"]: (
                0.0
                if not np.isfinite(
                    small_beta(m_axial_bare, diagonals[general.MASS["eta-AP"]][index])
                )
                else small_beta(m_axial_bare, diagonals[general.MASS["eta-AP"]][index])
            ),
            general.ANGLE["gamma"]: (
                0.0
                if not np.isfinite(
                    small_gamma(m_axial_bare, diagonals[general.MASS["etaP-AP"]][index])
                )
                else small_gamma(
                    m_axial_bare, diagonals[general.MASS["etaP-AP"]][index]
                )
            ),
        }
        if (
            previous is not None
            and np.abs(new_results["coupling"][previous] / x - 1) < proximity_thresh
        ):
            unknown_update = {k: new_results[k][previous] for k in unknown}
        elif old_results is not None:
            try:
                unknown_update = {
                    k: old_results[k][sort_args[index, 0]] for k in unknown
                }
            except IndexError:
                pass

        func = solver.create_func(
            system_of_equations,
            argument_order,
            known,
        )
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            for k, v in solver.solve_mixing(
                func,
                argument_order,
                unknown | unknown_update,
                abs_tolerance=tolerance,
            ).items():
                new_results[k][index] = v
            previous = index
            num_fails = 0
            successes.append(index)
        except (ValueError, RuntimeWarning):
            if num_fails is not None:
                num_fails += 1
        warnings.resetwarnings()
    for k in known:
        del new_results[k]
    for k, v in new_results.items():
        new_results[k] = v[successes]
    if old_results is not None:
        for k, v in old_results.items():
            new_results[k] = np.concatenate((v, new_results[k]))
    sort_results = np.argsort(new_results["coupling"], axis=-1)
    for k, v in new_results.items():
        new_results[k] = np.take_along_axis(v, sort_results, axis=-1)
    return new_results


def main():
    storage = pathlib.Path() / "results_axial.pkl"
    if storage.is_file():
        with open(storage, "rb") as f:
            old_results = pickle.load(f)
    else:
        old_results = {}

    for tol in [1e-8]:
        for quark in [(2, -1, -1), (1, 1, 1)]:
            # log spacing from 1e-6 to 1e6
            a,b = -6,6
            maxials = np.logspace(a, b, num=1 + int(10 * (b-a)))

            # fractional offsets from each meson mass
            a,b = -6,-0.5
            fracs = np.linspace(a, b, num=1 + int(10 * (b-a)))
            fracs = 1 + np.concatenate((-(10 ** fracs[::-1]), [0], 10**fracs))
            for m in [solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME]:
                maxials = np.concatenate((maxials, m * fracs))

            for maxial in maxials:
                # Don't redo initialization of maxial points
                if tol not in old_results:
                    old_results[tol] = {}
                if quark not in old_results[tol]:
                    old_results[tol][quark] = {}
                if not np.any(
                    np.abs(maxial / np.array(list(old_results[tol][quark].keys())) - 1)
                    <= 1e-13
                ):
                    print(tol, quark, maxial, "Initial")
                    a,b = -12,9
                    old_results[tol][quark][maxial] = scan_quarks(
                        np.logspace(a, b, num=1 + int(20 * (b-a))), tol, quark, maxial
                    )

                # Iteratively approach the numerical instability at high coupling strengths
                while True:
                    old_xs = np.log10(old_results[tol][quark][maxial]["coupling"])
                    print(tol, quark, maxial, 10 ** old_xs[-1])
                    new_xs = np.logspace(
                        old_xs.max(), old_xs.max() + np.diff(old_xs)[-1], num=101
                    )
                    old_results[tol][quark][maxial] = scan_quarks(
                        new_xs, tol, quark, maxial, old_results[tol][quark][maxial]
                    )
                    if not len(
                        set(np.log10(old_results[tol][quark][maxial]["coupling"]))
                        - set(old_xs)
                    ):
                        break

    if storage.is_file():
        storage.move(storage.with_suffix(".bak"))
    with open(storage, "wb") as f:
        pickle.dump(old_results, f)


# %%
if __name__ == "__main__":
    main()
