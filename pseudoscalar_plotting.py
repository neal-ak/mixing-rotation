# %%
""""""

# %%
import warnings

import numpy as np
import matplotlib.pyplot as plt

from rotation.symbolic.pseudo import create_mix_equation
from rotation.symbolic import general
from rotation.numerical import solver, pseudo

# %%
def scan_quarks(rotation_matrix, x_limits: tuple[float, float], quarks:tuple[float,float,float], m_phi_bare: float, g_chi=1, energy_pion=1, energy_kaon=1, num=11, tolerance=1e-8):
    x_points = np.logspace(*np.log10(x_limits[:2]), num=num)
    quarks = np.array(quarks[:3])
    equation = create_mix_equation(rotation_matrix)[0]
    argument_order, system_of_equations = solver.initialize_system(equation.lhs - equation.rhs)
    known = {
        general.MASS["pi0"]: solver.MASS_NEUTRAL_PION,
        general.MASS["eta"]: solver.MASS_ETA,
        general.MASS["etaP"]: solver.MASS_ETA_PRIME,
    }
    unknown = {
        general.MASS["pi0Bare"]: solver.MASS_NEUTRAL_PION,
        general.MASS["etaBare"]: solver.MASS_ETA,
        general.ANGLE["alpha"]: 0.0,
        general.ANGLE["beta"]: 0.0,
        general.ANGLE["gamma"]: 0.0,
        general.MASS["etaPBare"]: solver.MASS_ETA_PRIME,
        general.ANGLE["delta"]: 0.0,
        general.ANGLE["zeta"]: 0.0,
        general.ANGLE["kappa"]: 0.0,
    }

    known[general.MASS["phiBare"]] = m_phi_bare
    unknown[general.MASS["phi"]] = known[general.MASS["phiBare"]]

    saved_results = {k: np.nan*np.empty(x_points.shape) for k in argument_order}
    saved_results[general.COUPLING["chi"]] = g_chi * np.ones(x_points.shape)
    saved_results[general.PARAMETERS["Epipm"]] = energy_pion * np.ones(x_points.shape)
    saved_results[general.PARAMETERS["EKpm"]] = energy_kaon * np.ones(x_points.shape)

    warnings.filterwarnings("error", category=RuntimeWarning)
    errors = {}
    for index, x in enumerate(x_points):
        known |= pseudo.pseudo_calculate_bare_masses(*(quarks*x))
        for k,v in known.items():
            saved_results[k][index] = v
        func = solver.create_func(
            system_of_equations,
            argument_order,
            known,
        )
        try:
            for k,v in solver.solve_mixing(
                func,
                argument_order,
                unknown,
                abs_tolerance=tolerance,
            ).items():
                saved_results[k][index] = v
        except (ValueError, RuntimeWarning) as e:
            errors[x] = e
    warnings.resetwarnings()
    return x_points, saved_results, errors

# %%
def angle_plot(ax, xs, alpha, beta, gamma, delta=None, zeta=None, kappa=None):
    angles = {
        general.ANGLE["alpha"]: alpha,
        general.ANGLE["beta"]: beta,
        general.ANGLE["gamma"]: gamma,
        general.ANGLE["delta"]: delta,
        general.ANGLE["zeta"]: zeta,
        general.ANGLE["kappa"]: kappa,
    }
    for k,c in zip([general.ANGLE["alpha"], general.ANGLE["beta"], general.ANGLE["gamma"]], plt.get_cmap("tab10").colors[:3]):
        if angles[k] is None:
            continue
        ax.plot(xs, angles[k], color=c, label=f"${k}$", linestyle="-.")
    ylim = ax.get_ylim()
    for k,c in zip([general.ANGLE["delta"], general.ANGLE["zeta"], general.ANGLE["kappa"]], plt.get_cmap("tab10").colors[3:6]):
        if angles[k] is None:
            continue
        ax.plot(xs, angles[k], color=c, label=f"${k}$", linestyle="--")
    ax.set_ylim(ylim)
    ax.set_ylabel("Angle [Rad]")

    sec_y = ax.secondary_yaxis("right", functions=(lambda x: x*180/np.pi, lambda x: x*np.pi/180))
    sec_y.set_ylabel("Angle [Deg]")
    return sec_y

# %%
def mass_plot(ax, xs, phi_bare=None, phi=None, pi0_bare=None, pi0=None, eta_bare=None, eta=None, etaP_bare=None, etaP=None, fractional=False):
    masses = {
        "phi": (phi_bare, phi),
        "pi0": (pi0_bare, pi0),
        "eta": (eta_bare, eta),
        "etaP": (etaP_bare, etaP),
    }
    ax_sec = ax.twinx()
    colours = np.array(plt.get_cmap("tab20").colors)
    for (k,(bare,rotated)), c in zip(masses.items(), colours.reshape(-1,2,3)):
        col = c
        if k == "phi":
            col = col[::-1]
        if rotated is not None:
            ax.plot(xs, rotated, label=f"${general.MASS[k]}$", linestyle="--", color=col[1])
        if bare is not None:
            ax.plot(xs, bare, label=f"${general.MASS[k+'Bare']}$", linestyle=":", color=col[0])
        if bare is not None and rotated is not None:
            custom_linestyle = {"label": f"${general.MASS[k]}$/${general.MASS[k+'Bare']}$", "linestyle": "-.", "color": c[0]}
            compare = rotated/bare
            if fractional:
                custom_linestyle["label"] += "$-1$"
                compare -= 1
            ax_sec.plot(xs, compare, **custom_linestyle)
            ax.plot([], [], **custom_linestyle) # dummy plot to make the label appear on the original axis
    ax.set_ylabel("Mass [GeV]")
    if fractional:
        ax_sec.set_ylabel("(Rotated - Bare) / Bare")
    else:
        ax_sec.set_ylabel("Rotated to Bare Ratio")
    return ax_sec

# %%
def meson_and_gamma_plot(ax, xs, phi, pi0, eta, etaP, gamma, g_chi, m_phi, m_pi0_phi, m_eta_phi, m_etaP_phi, fractional=False):
    small_angle = {
        general.ANGLE["alpha"]: 0.5 * m_pi0_phi**2/(solver.MASS_NEUTRAL_PION**2 - m_phi**2),
        general.ANGLE["beta"]: 0.5 * m_eta_phi**2/(solver.MASS_ETA**2 - m_phi**2),
        general.ANGLE["gamma"]: 0.5 * m_etaP_phi**2/(solver.MASS_ETA_PRIME**2 - m_phi**2),
    }
    couples = {
        general.COUPLING["phi-chi"]: (phi, None),
        general.COUPLING["pi0-chi"]: (pi0, small_angle[general.ANGLE["alpha"]] * g_chi),
        general.COUPLING["eta-chi"]: (eta, small_angle[general.ANGLE["beta"]] * g_chi),
        general.COUPLING["etaP-chi"]: (etaP, small_angle[general.ANGLE["gamma"]] * g_chi),
        general.COUPLING["phi-gamma"]: (gamma, -solver.G_PI_GAMMA*small_angle[general.ANGLE["alpha"]] - solver.G_ETA_GAMMA*small_angle[general.ANGLE["beta"]] - solver.G_ETA_PRIME_GAMMA*small_angle[general.ANGLE["gamma"]]),
    }
    ax_sec = ax.twinx()
    for (label,(couple,approx)),colour in zip(couples.items(), plt.get_cmap("tab10").colors):
        if couple is not None:
            ax.plot(xs, couple, color=colour, label=f"${label}$", linestyle="--")
        if approx is not None:
            ax.plot(xs, approx, color=colour, label=f"${str(label)[:-1]},\\text{{approx}}}}$", linestyle=":")
        if couple is not None and approx is not None:
            custom_linestyle = {"color": colour, "label": f"${label} / {str(label)[:-1]},\\text{{approx}}}}$", "linestyle": "-."}
            compare = couple/approx
            if fractional:
                custom_linestyle["label"] += "$-1$"
                compare -= 1
            ax_sec.plot(xs, compare, **custom_linestyle)
            ax.plot([], [], **custom_linestyle)

    ax.set_ylabel("Coupling Strength")
    if fractional:
        ax_sec.set_ylabel("(Coupling - Approximation) / Approximation")
    else:
        ax_sec.set_ylabel("Coupling to Approximation Ratio")
    return ax_sec

# %%
def kaon_plot(ax, xs, phi, pi0, eta, etaP, m_phi, m_pi0_phi, m_eta_phi, m_etaP_phi, fractional=False):
    small_angle = {
        general.ANGLE["alpha"]: 0.5 * m_pi0_phi**2/(solver.MASS_NEUTRAL_PION**2 - m_phi**2),
        general.ANGLE["beta"]: 0.5 * m_eta_phi**2/(solver.MASS_ETA**2 - m_phi**2),
        general.ANGLE["gamma"]: 0.5 * m_etaP_phi**2/(solver.MASS_ETA_PRIME**2 - m_phi**2),
    }
    couples = {
        general.COUPLING["phi-pi-K"]: (phi, None),
        general.COUPLING["pi0-pi-K"]: (pi0, None),
        general.COUPLING["eta-pi-K"]: (eta, None),
        general.COUPLING["etaP-pi-K"]: (etaP, None),
    }
    ax_sec = ax.twinx()
    for (label,(couple,approx)),colour in zip(couples.items(), plt.get_cmap("tab10").colors[:4]):
        if couple is not None:
            ax.plot(xs, couple, color=colour, label=f"${label}$", linestyle="--")
        if approx is not None:
            ax.plot(xs, approx, color=colour, label=f"${str(label)[:-1]},\\text{{approx}}}}$", linestyle=":")
        if couple is not None and approx is not None:
            custom_linestyle = {"color": colour, "label": f"${label} / {str(label)[:-1]},\\text{{approx}}}}$", "linestyle": "-."}
            compare = couple/approx
            if fractional:
                custom_linestyle["label"] += "$-1$"
                compare -= 1
            ax_sec.plot(xs, compare, **custom_linestyle)
            ax.plot([], [], **custom_linestyle)

    ax.set_ylabel("Coupling Strength")
    if fractional:
        ax_sec.set_ylabel("(Coupling - Approximation) / Approximation")
    else:
        ax_sec.set_ylabel("Coupling to Approximation Ratio")
    return ax_sec

# %%
if __name__ == "__main__":
    rotate = general.create_4d_rotation()[0]
    x_limits, num, tolerance = (1e-6, 3.5), 101, 1e-13
    results = []
    for quarks in [(1,1,1), (2,-1,-1)]:
        for m_phi_bare in [1e-4, 1e-2, 1e0, 1e2]:
            results.append([*scan_quarks(rotate, x_limits, quarks, m_phi_bare, num=11, tolerance=1e-8), quarks, tolerance])

# %%
    for result in results:
        # for k in general.ANGLE.values():
        #     if k in result[1]:
        #         n = result[1][k]//(2*np.pi)
        #         result[1][k] -= 2 * np.pi * n
        result[1] |= pseudo.pseudo_calculate_meson_couplings(
            n_dim = rotate.rows,
            bare_mediator_dm_coupling = result[1][general.COUPLING["chi"]],
            alpha = result[1][general.ANGLE["alpha"]],
            beta = result[1][general.ANGLE["beta"]],
            gamma = result[1][general.ANGLE["gamma"]],
        )
        result[1][general.COUPLING["phi-gamma"]] = pseudo.pseudo_calculate_photon_coupling(
                n_dim = rotate.rows,
                alpha = result[1][general.ANGLE["alpha"]],
                beta = result[1][general.ANGLE["beta"]],
                gamma = result[1][general.ANGLE["gamma"]],
                delta = result[1][general.ANGLE["delta"]],
                zeta = result[1][general.ANGLE["zeta"]],
                kappa = result[1][general.ANGLE["kappa"]],
            )
        result[1] |= pseudo.pseudo_calculate_bare_kaon(
            energy_pion=result[1][general.PARAMETERS["Epipm"]],
            energy_kaon=result[1][general.PARAMETERS["EKpm"]],
        )
        result[1] |= pseudo.pseudo_calculate_kaon_couplings(
            n_dim=rotate.rows,
            bare_mediator_pion_kaon_coupling=result[1][general.COUPLING["phiBare-pi-K"]],
            bare_pion_pion_kaon_coupling=result[1][general.COUPLING["pi0Bare-pi-K"]],
            bare_eta_pion_kaon_coupling=result[1][general.COUPLING["etaBare-pi-K"]],
            bare_etaP_pion_kaon_coupling=result[1][general.COUPLING["etaPBare-pi-K"]],
            alpha = result[1][general.ANGLE["alpha"]],
            beta = result[1][general.ANGLE["beta"]],
            gamma = result[1][general.ANGLE["gamma"]],
            delta = result[1][general.ANGLE["delta"]],
            zeta = result[1][general.ANGLE["zeta"]],
            kappa = result[1][general.ANGLE["kappa"]],
        )

# %%
    # ratio vs fractional error: ratio better for extreme differences, fractional better for smaller
    # Plot sets for: Heavy, Light, and one near a meson mass
    for result in results:
        fig, axs = plt.subplots(2,2, figsize=(15,15))
        nan = np.isnan(result[1][general.ANGLE["alpha"]])
        ax_degree = angle_plot(
            axs[0,0],
            result[0][~nan],
            *[result[1][general.ANGLE[k]][~nan] for k in [
                "alpha",
                "beta",
                "gamma",
                "delta",
                "zeta",
                "kappa"
            ]],
        )
        axs[0,0].set_title("Rotation Angles")
        ax_mass = mass_plot(
            axs[0,1],
            result[0][~nan],
            phi_bare=result[1][general.MASS["phiBare"]][~nan],
            phi=result[1][general.MASS["phi"]][~nan],
            pi0_bare=result[1][general.MASS["pi0Bare"]][~nan],
            pi0=result[1][general.MASS["pi0"]][~nan],
            eta_bare=result[1][general.MASS["etaBare"]][~nan],
            eta=result[1][general.MASS["eta"]][~nan],
            etaP_bare=result[1][general.MASS["etaPBare"]][~nan],
            etaP=result[1][general.MASS["etaP"]][~nan],
            # fractional=True,
        )
        axs[0,1].set_title("Particle Masses")
        ax_meson = meson_and_gamma_plot(
            axs[1,0],
            result[0][~nan],
            phi=result[1][general.COUPLING["phi-chi"]][~nan],
            pi0=result[1][general.COUPLING["pi0-chi"]][~nan],
            eta=result[1][general.COUPLING["eta-chi"]][~nan],
            etaP=result[1][general.COUPLING["etaP-chi"]][~nan],
            gamma=result[1][general.COUPLING["phi-gamma"]][~nan],
            g_chi=result[1][general.COUPLING["chi"]][~nan],
            m_phi=result[1][general.MASS["phi"]][~nan],
            m_pi0_phi=result[1][general.MASS["pi0-phi"]][~nan],
            m_eta_phi=result[1][general.MASS["eta-phi"]][~nan],
            m_etaP_phi=result[1][general.MASS["etaP-phi"]][~nan],
            # fractional=True,
        )
        axs[1,0].set_title(f"Meson and Gamma Couplings: ${general.COUPLING["chi"]} = {result[1][general.COUPLING["chi"]][0]}$")
        # ax_meson.set_ylim(-1,1)
        ax_kaon = kaon_plot(
            axs[1,1],
            result[0][~nan],
            phi=result[1][general.COUPLING["phi-pi-K"]][~nan],
            pi0=result[1][general.COUPLING["pi0-pi-K"]][~nan],
            eta=result[1][general.COUPLING["eta-pi-K"]][~nan],
            etaP=result[1][general.COUPLING["etaP-pi-K"]][~nan],
            m_phi=result[1][general.MASS["phi"]][~nan],
            m_pi0_phi=result[1][general.MASS["pi0-phi"]][~nan],
            m_eta_phi=result[1][general.MASS["eta-phi"]][~nan],
            m_etaP_phi=result[1][general.MASS["etaP-phi"]][~nan],
            # fractional=True,
        )
        axs[1,1].set_title("Kaon Couplings: " + ", ".join([f"${general.PARAMETERS[k]} = {result[1][general.PARAMETERS[k]][0]}$ GeV" for k in ["Epipm", "EKpm"]]))
        for ax in axs.flat:
            ax.legend()
            ax.set_xlabel(f"$\\text{{Quark coupling strength: (}}{',\\quad'.join([f'{general.COUPLING[s]}={p}' for s,p in zip('uds',result[3])])} \\text{{)}}$")
        fig.suptitle(f"${general.MASS['phiBare']} = {result[1][general.MASS['phiBare']][0]}$ GeV, tol$={result[4]}$")
        fig.tight_layout()
        fig.show()
