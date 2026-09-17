# %%
""""""

# %%
import pathlib
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt

from rotation.symbolic.pseudo import create_mix_equation
from rotation.symbolic import general
from rotation.numerical import solver, pseudo

def small_alpha(m_phi, m_pi0_phi):
    return 0.5 * m_pi0_phi**2/(solver.MASS_NEUTRAL_PION**2 - m_phi**2)

def small_beta(m_phi, m_eta_phi):
    return 0.5 * m_eta_phi**2/(solver.MASS_ETA**2 - m_phi**2)

def small_gamma(m_phi, m_etaP_phi):
    return 0.5 * m_etaP_phi**2/(solver.MASS_ETA_PRIME**2 - m_phi**2)

# %%
def scan_quarks(rotation_matrix, x_limits: tuple[float, float], quarks: tuple[float,float,float], m_phi_bare: float, g_chi=1, energy_pion=1, energy_kaon=1, num=11, tolerance=1e-8):
    x_points = np.logspace(*np.log10(x_limits[:2]), num=num)
    quarks = np.array(quarks[:3])
    equation = create_mix_equation(rotation_matrix)[0]
    argument_order, system_of_equations = solver.initialize_system(equation.lhs - equation.rhs)

    saved_results = {k: np.nan*np.empty(x_points.shape) for k in argument_order}
    saved_results[general.MASS["phiBare"]] = m_phi_bare * np.ones(x_points.shape)
    saved_results[general.MASS["pi0"]] = solver.MASS_NEUTRAL_PION * np.ones(x_points.shape)
    saved_results[general.MASS["eta"]] = solver.MASS_ETA * np.ones(x_points.shape)
    saved_results[general.MASS["etaP"]] = solver.MASS_ETA_PRIME * np.ones(x_points.shape)
    saved_results |= pseudo.pseudo_calculate_bare_masses(*(x_points*quarks[:,np.newaxis]))
    saved_results[general.COUPLING["chi"]] = g_chi * np.ones(x_points.shape)
    saved_results[general.PARAMETERS["Epipm"]] = energy_pion * np.ones(x_points.shape)
    saved_results[general.PARAMETERS["EKpm"]] = energy_kaon * np.ones(x_points.shape)

    knowns = [
        {
            general.MASS["phiBare"]: saved_results[general.MASS["phiBare"]][0],
            general.MASS["pi0"]: saved_results[general.MASS["pi0"]][0],
            general.MASS["eta"]: saved_results[general.MASS["eta"]][0],
            general.MASS["etaP"]: saved_results[general.MASS["etaP"]][0],
            general.MASS["pi0-phi"]: mpp,
            general.MASS["eta-phi"]: mep,
            general.MASS["etaP-phi"]: mepp,
        } for mpp, mep, mepp in zip(saved_results[general.MASS["pi0-phi"]], saved_results[general.MASS["eta-phi"]], saved_results[general.MASS["etaP-phi"]])
    ]
    unknowns = [{
        general.MASS["phi"]: know[general.MASS["phiBare"]],
        general.MASS["pi0Bare"]: know[general.MASS["pi0"]],
        general.MASS["etaBare"]: know[general.MASS["eta"]],
        general.MASS["etaPBare"]: know[general.MASS["etaP"]],
        general.ANGLE["alpha"]: 0.0 if not np.isfinite(small_alpha(know[general.MASS["phiBare"]], know[general.MASS["pi0-phi"]])) else small_alpha(know[general.MASS["phiBare"]], know[general.MASS["pi0-phi"]]),
        general.ANGLE["beta"]: 0.0 if not np.isfinite(small_beta(know[general.MASS["phiBare"]], know[general.MASS["eta-phi"]])) else small_beta(know[general.MASS["phiBare"]], know[general.MASS["eta-phi"]]),
        general.ANGLE["gamma"]: 0.0 if not np.isfinite(small_gamma(know[general.MASS["phiBare"]], know[general.MASS["etaP-phi"]])) else small_gamma(know[general.MASS["phiBare"]], know[general.MASS["etaP-phi"]]),
        general.ANGLE["delta"]: 0.0,
        general.ANGLE["zeta"]: 0.0,
        general.ANGLE["kappa"]: 0.0,
    } for know in knowns]

    warnings.filterwarnings("error", category=RuntimeWarning)
    errors = {}
    previous = None
    num_fails = None
    for index, (know, unknow) in enumerate(zip(knowns, unknowns)):
        if num_fails is not None and num_fails > len(x_points)/4:
            break
        if previous is not None:
            for k in unknow:
                if k in saved_results:
                    unknow[k] = saved_results[k][previous]
        func = solver.create_func(
            system_of_equations,
            argument_order,
            know,
        )
        try:
            for k,v in solver.solve_mixing(
                func,
                argument_order,
                unknow,
                abs_tolerance=tolerance,
            ).items():
                saved_results[k][index] = v
            previous = index
            num_fails = 0
        except (ValueError, RuntimeWarning) as e:
            if num_fails is not None:
                num_fails += 1
            errors[x_points[index]] = e
    warnings.resetwarnings()
    return x_points, saved_results, errors

# %%
def angle_plot(ax, xs, alpha, beta, gamma, delta=None, zeta=None, kappa=None, m_phi=None, m_pi0_phi=None, m_eta_phi=None, m_etaP_phi=None):
    angles = {
        general.ANGLE["alpha"]: alpha,
        general.ANGLE["beta"]: beta,
        general.ANGLE["gamma"]: gamma,
        general.ANGLE["delta"]: delta,
        general.ANGLE["zeta"]: zeta,
        general.ANGLE["kappa"]: kappa,
    }
    smalls = {
        general.ANGLE["alpha"]: None if m_phi is None or m_pi0_phi is None else small_alpha(m_phi, m_pi0_phi),
        general.ANGLE["beta"]: None if m_phi is None or m_eta_phi is None else small_beta(m_phi, m_eta_phi),
        general.ANGLE["gamma"]: None if m_phi is None or m_etaP_phi is None else small_gamma(m_phi, m_etaP_phi),
        general.ANGLE["delta"]: None,
        general.ANGLE["zeta"]: None,
        general.ANGLE["kappa"]: None,
    }
    for k,c in zip([general.ANGLE["alpha"], general.ANGLE["beta"], general.ANGLE["gamma"]], plt.get_cmap("tab10").colors[:3]):
        if angles[k] is not None:
            ax.plot(xs, angles[k], color=c, label=f"${k}$", linestyle="--", marker="")
        if smalls[k] is not None:
            ax.plot(xs, smalls[k], color=c, label=f"${k}_{{\\text{{small}}}}$", linestyle=":", marker="")
    ylim = ax.get_ylim()
    for k,c in zip([general.ANGLE["delta"], general.ANGLE["zeta"], general.ANGLE["kappa"]], plt.get_cmap("tab10").colors[3:6]):
        if angles[k] is None:
            continue
        ax.plot(xs, angles[k], color=c, label=f"${k}$", linestyle="--", marker="")
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
    for (k,(bare,rotated)), c in zip(masses.items(), ["black"]+list(colours.reshape(-1,2,3))):
        col = c
        if k == "phi":
            continue
            col = col[::-1]
        if rotated is not None:
            ax.plot(xs, rotated, label=f"${general.MASS[k]}$", linestyle="--", color=col[1], marker="")
        if bare is not None:
            ax.plot(xs, bare, label=f"${general.MASS[k+'Bare']}$", linestyle=":", color=col[0], marker="")
        if bare is not None and rotated is not None:
            custom_linestyle = {"label": f"${general.MASS[k]}$/${general.MASS[k+'Bare']}$", "linestyle": "-.", "color": c[0], "marker":""}
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
    couples = {
        general.COUPLING["phi-chi"]: (phi, None),
        general.COUPLING["pi0-chi"]: (pi0, small_alpha(m_phi, m_pi0_phi) * g_chi),
        general.COUPLING["eta-chi"]: (eta, small_beta(m_phi, m_eta_phi) * g_chi),
        general.COUPLING["etaP-chi"]: (etaP, small_gamma(m_phi, m_etaP_phi) * g_chi),
        general.COUPLING["phi-gamma"]: (gamma, -solver.G_PI_GAMMA*small_alpha(m_phi, m_pi0_phi) - solver.G_ETA_GAMMA*small_beta(m_phi, m_eta_phi) - solver.G_ETA_PRIME_GAMMA*small_gamma(m_phi, m_etaP_phi)),
    }
    ax_sec = ax.twinx()
    for (label,(couple,approx)),colour in zip(couples.items(), ["black"]+list(plt.get_cmap("tab10").colors)):
        if couple is not None:
            ax.plot(xs, couple, color=colour, label=f"${label}$", linestyle="--", marker="")
        if approx is not None:
            ax.plot(xs, approx, color=colour, label=f"${str(label)[:-1]},\\text{{approx}}}}$", linestyle=":", marker="")
        if couple is not None and approx is not None:
            custom_linestyle = {"color": colour, "label": f"squared: ${label} / {str(label)[:-1]},\\text{{approx}}}}$", "linestyle": "-.", "marker":""}
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
    couples = {
        general.COUPLING["phi-pi-K"]: (phi, None),
        general.COUPLING["pi0-pi-K"]: (pi0, None),
        general.COUPLING["eta-pi-K"]: (eta, None),
        general.COUPLING["etaP-pi-K"]: (etaP, None),
    }
    ax_sec = ax.twinx()
    for (label,(couple,approx)),colour in zip(couples.items(), ["black"]+list(plt.get_cmap("tab10").colors[:3])):
        if couple is not None:
            ax.plot(xs, couple, color=colour, label=f"${label}$", linestyle="--", marker="")
        if approx is not None:
            ax.plot(xs, approx, color=colour, label=f"${str(label)[:-1]},\\text{{approx}}}}$", linestyle=":", marker="")
        if couple is not None and approx is not None:
            custom_linestyle = {"color": colour, "label": f"${label} / {str(label)[:-1]},\\text{{approx}}}}$", "linestyle": "-.", "marker":""}
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
    storage = pathlib.Path() / "results_pseudo_tol6_hist.pkl"
    if storage.is_file():
        print(f"Loading results from {storage}")
        with open(storage, "rb") as f:
            results = pickle.load(f)
    else:
        print(f"Solving... this will take a bit...")
        rotate = general.create_4d_rotation()[0]
        num, tolerance = 601, 1e-6
        mass_limits = {
            1e-3: (1e-4, 1e-1),
            1e-2: (1e-4, 1e-1),
            solver.MASS_NEUTRAL_PION*0.9: (1e-4, 3e-1),
            solver.MASS_NEUTRAL_PION*0.99: (1e-4, 3e-1),
            solver.MASS_NEUTRAL_PION*0.999: (1e-5, 3e-1),
            solver.MASS_NEUTRAL_PION: (1e-6, 3e-1),
            solver.MASS_NEUTRAL_PION*1.001: (1e-5, 3e-1),
            solver.MASS_NEUTRAL_PION*1.01: (1e-4, 3e-1),
            solver.MASS_NEUTRAL_PION*1.1: (1e-4, 3e-1),
            solver.MASS_ETA*0.9: (1e-3, 1e0),
            solver.MASS_ETA*0.99: (1e-3, 1e0),
            solver.MASS_ETA*0.999: (1e-4, 1e0),
            solver.MASS_ETA: (1e-5, 1e0),
            solver.MASS_ETA*1.001: (1e-4, 1e0),
            solver.MASS_ETA*1.01: (1e-3, 1e0),
            solver.MASS_ETA*1.1: (1e-3, 1e0),
            solver.MASS_ETA_PRIME*0.9: (1e-3, 1e0),
            solver.MASS_ETA_PRIME*0.99: (1e-3, 1e0),
            solver.MASS_ETA_PRIME*0.999: (1e-3, 1e0),
            solver.MASS_ETA_PRIME: (1e-4, 1e0),
            solver.MASS_ETA_PRIME*1.001: (1e-3, 1e0),
            solver.MASS_ETA_PRIME*1.01: (1e-3, 1e0),
            solver.MASS_ETA_PRIME*1.1: (1e-3, 1e0),
            1e1: (1e-2, 3e0),
            1e2: (1e-1, 1e1),
            1e3: (1e-1, 3e1),
        }
        results = []
        for quarks in [(1,1,1), (2,-1,-1)]:
            for m_phi_bare, x_limits in mass_limits.items():
                # if not quarks == (1,1,1) or not m_phi_bare == solver.MASS_ETA_PRIME:
                #     continue
                print(quarks, m_phi_bare)
                results.append([*scan_quarks(rotate, x_limits, quarks, m_phi_bare, num=num, tolerance=tolerance), quarks, tolerance])

        for result in results:
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
        print(f"Storing results in {storage}")
        with open(storage, "wb") as f:
            pickle.dump(results, f)

# %%
    for result in results:
        # if not result[3] == (1,1,1) or not np.all(result[1][general.MASS["phiBare"]] == solver.MASS_ETA_PRIME):
        #     continue
        nan = np.isnan(result[1][general.ANGLE["alpha"]])
        fig, axs = plt.subplots(2,2, figsize=(15,15), sharex=True)
        fig.suptitle(f"${general.MASS['phiBare']} = {result[1][general.MASS['phiBare']][0]}$ GeV, tol$={result[4]}$")

        axs[0,0].set_title("Rotation Angles")
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
            m_phi=result[1][general.MASS["phiBare"]][~nan],
            m_pi0_phi=result[1][general.MASS["pi0-phi"]][~nan],
            m_eta_phi=result[1][general.MASS["eta-phi"]][~nan],
            m_etaP_phi=result[1][general.MASS["etaP-phi"]][~nan],
        )

        axs[0,1].set_title("Particle Masses")
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

        axs[1,0].set_title(f"Meson and Gamma Couplings: ${general.COUPLING["chi"]} = {result[1][general.COUPLING["chi"]][0]}$")
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

        axs[1,1].set_title("Kaon Couplings: " + ", ".join([f"${general.PARAMETERS[k]} = {result[1][general.PARAMETERS[k]][0]}$ GeV" for k in ["Epipm", "EKpm"]]))
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

        for ax in axs.flat:
            ax.legend()
            ax.set_xlabel(f"$\\text{{Quark coupling strength: (}}{',\\quad'.join([f'{general.COUPLING[s]}={p}' for s,p in zip('uds',result[3])])} \\text{{)}}$")
            ax.set_xscale("symlog", linthresh=result[0][~nan].min()*1e-2)
            # xlims = ax.get_xlim()
            # if xlims[0] < 1e-7:
            #     ax.set_xlim(1e-6, xlims[1])
            ax.set_xlim(result[0].min(), result[0].max())

        fig.tight_layout()
        if result[3] == (1,1,1):
            file_name = "equal"
        elif result[3] == (2,-1,-1):
            file_name = "charge"
        else:
            file_name = "(" + "_".join([f"{_x:.1e}" for _x in result[3]]) + ")"
        file_name += f"_{result[1][solver.MASS["phiBare"]][0]:.4e}"
        fig.savefig(f"pseudo_{file_name}_hist.pdf")
        # fig.show()
