# %%
import pathlib
import pickle

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

from rotation.numerical import solver, pseudo
from rotation.symbolic import general

def small_alpha(m_phi, m_pi0_phi):
    return 0.5 * m_pi0_phi**2/(solver.MASS_NEUTRAL_PION**2 - m_phi**2)

def small_beta(m_phi, m_eta_phi):
    return 0.5 * m_eta_phi**2/(solver.MASS_ETA**2 - m_phi**2)

def small_gamma(m_phi, m_etaP_phi):
    return 0.5 * m_etaP_phi**2/(solver.MASS_ETA_PRIME**2 - m_phi**2)

# %%
if __name__ == "__main__":
    stored_results = pathlib.Path() / "results_pseudo.pkl"
    if stored_results.is_file():
        with open(stored_results, "rb") as f:
            stored = pickle.load(f)
    else:
        stored = {}

# %%
    for tolerance in stored:
        for quarks, stored[tolerance][quarks] in stored[tolerance].items():
            phis = sorted(stored[tolerance][quarks])
            stored[tolerance][quarks] = {k: v[np.newaxis, ...] for k,v in stored[tolerance][quarks][phis[0]].items()}
            for p in phis[1:]:
                syn_shape = next(iter(stored[tolerance][quarks].values())).shape
                curr_shape = next(iter(stored[tolerance][quarks][p].values())).shape
                if curr_shape[-1] > syn_shape[-1]:
                    for k,v in stored[tolerance][quarks].items():
                        stored[tolerance][quarks][k] = np.pad(v, ((0,0),(0,curr_shape[-1]-syn_shape[-1])), mode="constant", constant_values=np.nan)
                    syn_shape = *syn_shape[:-1], curr_shape[-1]
                for k,v in stored[tolerance][quarks][p].items():
                    if k in stored[tolerance][quarks]:
                        if syn_shape[-1] > curr_shape[-1]:
                            v = np.pad(v, (0,syn_shape[-1]-curr_shape[-1]), mode="constant", constant_values=np.nan)
                        stored[tolerance][quarks][k] = np.append(stored[tolerance][quarks][k], v[np.newaxis, ...], axis=0)
                    else:
                        raise ValueError("Had an extra key not in others, they should all be the same.")
            stored[tolerance][quarks][general.MASS["phiBare"]] = np.array(phis)

    g_chi = 1
    approx, ratio, fraction = {}, {}, {}
    for tolerance in stored:
        approx[tolerance], ratio[tolerance], fraction[tolerance] = {}, {}, {}
        for quarks in stored[tolerance]:
            stored[tolerance][quarks] |= pseudo.pseudo_calculate_meson_couplings(
                n_dim = 4,
                bare_mediator_dm_coupling = g_chi,
                alpha = stored[tolerance][quarks][general.ANGLE["alpha"]],
                beta = stored[tolerance][quarks][general.ANGLE["beta"]],
                gamma = stored[tolerance][quarks][general.ANGLE["gamma"]],
            )
            stored[tolerance][quarks][general.COUPLING["phi-gamma"]] = pseudo.pseudo_calculate_photon_coupling(
                n_dim = 4,
                alpha = stored[tolerance][quarks][general.ANGLE["alpha"]],
                beta = stored[tolerance][quarks][general.ANGLE["beta"]],
                gamma = stored[tolerance][quarks][general.ANGLE["gamma"]],
                delta = stored[tolerance][quarks][general.ANGLE["delta"]],
                zeta = stored[tolerance][quarks][general.ANGLE["zeta"]],
                kappa = stored[tolerance][quarks][general.ANGLE["kappa"]],
            )
            m_diag = pseudo.pseudo_calculate_bare_masses(*(stored[tolerance][quarks]["coupling"]*np.array(quarks)[:,np.newaxis,np.newaxis]))
            approx[tolerance][quarks] = {general.ANGLE[a]: f(stored[tolerance][quarks][general.MASS["phi"]], m_diag[general.MASS[m+"-phi"]]) for a, f, m in zip(("alpha", "beta", "gamma"), (small_alpha, small_beta, small_gamma), ("pi0", "eta", "etaP"))}
            approx[tolerance][quarks] |= {
                general.COUPLING["pi0-chi"]: approx[tolerance][quarks][general.ANGLE["alpha"]] * g_chi,
                general.COUPLING["eta-chi"]: approx[tolerance][quarks][general.ANGLE["beta"]] * g_chi,
                general.COUPLING["etaP-chi"]: approx[tolerance][quarks][general.ANGLE["gamma"]] * g_chi,
                general.COUPLING["phi-gamma"]: -solver.G_PI_GAMMA*approx[tolerance][quarks][general.ANGLE["alpha"]] - solver.G_ETA_GAMMA*approx[tolerance][quarks][general.ANGLE["beta"]] - solver.G_ETA_PRIME_GAMMA*approx[tolerance][quarks][general.ANGLE["gamma"]],
            }
            ratio[tolerance][quarks] = {k: v/stored[tolerance][quarks][k] for k,v in approx[tolerance][quarks].items()}
            ratio[tolerance][quarks] |= {
                general.MASS["phi"]: stored[tolerance][quarks][general.MASS["phi"]] / stored[tolerance][quarks][general.MASS["phiBare"]][...,np.newaxis],
                general.MASS["pi0Bare"]: solver.MASS_NEUTRAL_PION / stored[tolerance][quarks][general.MASS["pi0Bare"]],
                general.MASS["etaBare"]: solver.MASS_ETA / stored[tolerance][quarks][general.MASS["etaBare"]],
                general.MASS["etaPBare"]: solver.MASS_ETA_PRIME / stored[tolerance][quarks][general.MASS["etaPBare"]],
            }
            fraction[tolerance][quarks] = {k: np.abs(v-1) for k,v in ratio[tolerance][quarks].items()}

# %%
    tab20 = np.array(plt.get_cmap("tab20").colors).reshape(10,2,3)
    angle_style = dict(zip([general.ANGLE[k] for k in [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "zeta",
        "kappa",
    ]], tab20))
    mass_style = dict(zip([general.MASS[k] for k in ["phi", "pi0Bare", "etaBare", "etaPBare"]], tab20[[7,0,1,2]]))
    meson_style = dict(zip([general.COUPLING[k] for k in ["phi-chi", "pi0-chi", "eta-chi", "etaP-chi", "phi-gamma"]], tab20[[7,0,1,2,6]]))
    g_chi = 1
    fractional = np.array((True, False, True), dtype=bool)
    height, width, figsize = 2, 2, 4
    for tolerance in [1e-8]:
        for quarks in [(2,-1,-1)]:
            phB, xs = stored[tolerance][quarks][general.MASS["phiBare"]], stored[tolerance][quarks]["coupling"]
            masks = {
                "other": (np.log10(phB)-np.log10(phB).min()) % np.diff(np.log10(phB))[0] < 1e-6,
                "pi": np.abs(phB/solver.MASS_NEUTRAL_PION - 1) < 2e-1,
                "eta": np.abs(phB/solver.MASS_ETA - 1) < 2e-1,
                "etaP": np.abs(phB/solver.MASS_ETA_PRIME - 1) < 2e-1,
            }
            erratic_thresh = 1e-4
            erratic = (np.abs(phB/solver.MASS_NEUTRAL_PION-1)<=erratic_thresh) | (np.abs(phB/solver.MASS_ETA-1)<=erratic_thresh) | (np.abs(phB/solver.MASS_ETA_PRIME-1)<=erratic_thresh)

            for mask_name, mask in masks.items():
                fig, axs = plt.subplots(height,width, squeeze=False, figsize=figsize*np.array((width,height)), sharex=True)
                limits = np.empty((*axs.shape, 2))

                # Prep limits
                # Angles
                ax = axs[0,0]
                ax.set_xscale("log")
                for label in angle_style:
                    ax.plot(xs[mask&~erratic].T, stored[tolerance][quarks][label][mask&~erratic].T)
                    if label in approx:
                        ax.plot(xs[mask&~erratic].T, approx[tolerance][quarks][label][mask&~erratic].T)
                angle_limits = (
                    max(ax.get_ylim()[0], -1/2*np.pi),
                    min(ax.get_ylim()[1], 1/2*np.pi),
                )
                # Masses
                ax = axs[0,1]
                ax.set_xscale("log")
                for label in mass_style:
                    ax.plot(xs[mask&~erratic].T, stored[tolerance][quarks][label][mask&~erratic].T)
                    if label in approx:
                        ax.plot(xs[mask&~erratic].T, approx[tolerance][quarks][label][mask&~erratic].T)
                mass_limits = (
                    max(ax.get_ylim()[0], 1e-3),
                    ax.get_ylim()[1],
                )
                # Meson couplings
                ax = axs[1,0]
                ax.set_xscale("log")
                for label in meson_style:
                    ax.plot(xs[mask&~erratic].T, stored[tolerance][quarks][label][mask&~erratic].T)
                    if label in approx:
                        ax.plot(xs[mask&~erratic].T, approx[tolerance][quarks][label][mask&~erratic].T)
                meson_limits = (
                    max(ax.get_ylim()[0], -10*g_chi),
                    min(ax.get_ylim()[1], 10*g_chi),
                )
                # # Ratio
                # ax = axs[1,1]
                # ax.set_xscale("log")
                # for label in angle_style|mass_style|meson_style:
                #     if label in ratio[tolerance][quarks]:
                #         ax.plot(xs[mask&~erratic].T, ratio[tolerance][quarks][label][mask&~erratic].T)
                # ratio_limits = (
                #     max(ax.get_ylim()[0], -10),
                #     min(ax.get_ylim()[1], 10),
                # )
                # Fractional error
                ax = axs[1,1]
                ax.set_xscale("log")
                for label in angle_style|mass_style|meson_style:
                    if label in fraction[tolerance][quarks]:
                        ax.plot(xs[mask&~erratic].T, fraction[tolerance][quarks][label][mask&~erratic].T)
                fraction_limits = (
                    max(ax.get_ylim()[0], -1),
                    min(ax.get_ylim()[1], 1),
                )
                plt.close()

                fig, axs = plt.subplots(height,width, squeeze=False, figsize=figsize*np.array((width,height)), sharex=True)
                for ax in axs[-1,:]:
                    ax.set_xlabel(f"$\\text{{Quark coupling strength: (}}{',\\quad'.join([f'{general.COUPLING[s]}={p}' for s,p in zip('uds',quarks)])} \\text{{)}}$")
                    ax.set_xlim(1e-7, 1e1)
                    ax.set_xscale("log")

                plot_elements = []

                # Angles
                ax = axs[0,0]
                for label, colours in angle_style.items():
                    line = ax.plot(xs[mask][0], stored[tolerance][quarks][label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.", marker="")
                    plot_elements.append((line[0], xs[mask], stored[tolerance][quarks][label][mask]))
                    if label in approx[tolerance][quarks]:
                        line = ax.plot(xs[mask][0], approx[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(approx[tolerance][quarks][label][mask])) else ""+f"${label}_{{\\text{{approx}}}}$", color=colours[1], linewidth=1, marker="")
                        plot_elements.append((line[0], xs[mask], approx[tolerance][quarks][label][mask]))
                ax.set(
                    ylim=angle_limits,
                    ylabel="Angle [Rad]",
                )
                ax.legend(loc="lower left")
                sec_y = ax.secondary_yaxis("right", functions=(lambda x: x*180/np.pi, lambda x: x*np.pi/180))
                sec_y.set_ylabel("Angle [Deg]")

                # Masses
                ax = axs[0,1]
                phi_hline = ax.axhline(phB[0], color="black", linewidth=1)
                plot_elements.append((phi_hline, None, phB[...,np.newaxis]))
                for label, colours in mass_style.items():
                    line = ax.plot(xs[mask][0], stored[tolerance][quarks][label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.", marker="")
                    plot_elements.append((line[0], xs[mask], stored[tolerance][quarks][label][mask]))
                for m, colours in zip((solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME), tab20):
                    ax.axhline(m, color=colours[1], linewidth=1)
                ax.set(
                    ylim=mass_limits,
                    ylabel="Mass [GeV]",
                    yscale="log",
                )
                ax.legend(loc="lower left")

                # Couplings
                ax = axs[1,0]
                ax.axhline(g_chi, color=tab20[7,1], linewidth=1, label=f"${general.COUPLING['chi']} = {g_chi}$")
                for label, colours in meson_style.items():
                    line = ax.plot(xs[mask][0], stored[tolerance][quarks][label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.", marker="")
                    plot_elements.append((line[0], xs[mask], stored[tolerance][quarks][label][mask]))
                    if label in approx[tolerance][quarks]:
                        line = ax.plot(xs[mask][0], approx[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(approx[tolerance][quarks][label][mask])) else ""+f"${str(label)[:-1]},\\text{{approx}}}}$", color=colours[1], linewidth=1, marker="")
                        plot_elements.append((line[0], xs[mask], approx[tolerance][quarks][label][mask]))
                ax.set(
                    ylim=meson_limits,
                    ylabel="Coupling",
                )
                ax.legend(loc="lower left")

                # # Ratio
                # ax = axs[1,1]
                # for label, colours in angle_style.items():
                #     if label in ratio[tolerance][quarks]:
                #         line = ax.plot(xs[mask][0], ratio[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(ratio[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle="--", linewidth=1, marker="")
                #         plot_elements.append((line[0], xs[mask], ratio[tolerance][quarks][label][mask]))
                # for label, colours in mass_style.items():
                #     if label in ratio[tolerance][quarks]:
                #         line = ax.plot(xs[mask][0], ratio[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(ratio[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle="-.", linewidth=1, marker="")
                #         plot_elements.append((line[0], xs[mask], ratio[tolerance][quarks][label][mask]))
                # for label, colours in meson_style.items():
                #     if label in ratio[tolerance][quarks]:
                #         line = ax.plot(xs[mask][0], ratio[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(ratio[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle=":", linewidth=1, marker="")
                #         plot_elements.append((line[0], xs[mask], ratio[tolerance][quarks][label][mask]))
                # ax.set(
                #     ylim=ratio_limits,
                #     ylabel="Approx / Reference",
                # )
                # ax.legend(loc="lower left")

                # Fractional error
                ax = axs[1,1]
                for label, colours in angle_style.items():
                    if label in fraction[tolerance][quarks]:
                        line = ax.plot(xs[mask][0], fraction[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(fraction[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle="--", linewidth=1, marker="")
                        plot_elements.append((line[0], xs[mask], fraction[tolerance][quarks][label][mask]))
                for label, colours in mass_style.items():
                    if label in fraction[tolerance][quarks]:
                        line = ax.plot(xs[mask][0], fraction[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(fraction[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle="-.", linewidth=1, marker="")
                        plot_elements.append((line[0], xs[mask], fraction[tolerance][quarks][label][mask]))
                for label, colours in meson_style.items():
                    if label in fraction[tolerance][quarks]:
                        line = ax.plot(xs[mask][0], fraction[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(fraction[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle=":", linewidth=1, marker="")
                        plot_elements.append((line[0], xs[mask], fraction[tolerance][quarks][label][mask]))
                ax.set(
                    ylim=fraction_limits,
                    ylabel="| Approx / Reference - 1 |",
                )
                ax.legend(loc="lower left")

                for ax in axs.flat:
                    ax.grid()
                    if ax.get_yscale() == "log":
                        ax.set_ylim(
                            10 ** (np.log10(ax.get_ylim())[0] - 0.05 * np.diff(np.log10(ax.get_ylim()))[0]),
                            10 ** (np.log10(ax.get_ylim())[1] + 0.05 * np.diff(np.log10(ax.get_ylim()))[0]),
                        )
                    elif ax.get_yscale() == "linear":
                        ax.set_ylim(
                            ax.get_ylim()[0] - 0.05 * np.diff(ax.get_ylim())[0],
                            ax.get_ylim()[1] + 0.05 * np.diff(ax.get_ylim())[0],
                        )

                fig.suptitle(f"${general.MASS['phiBare']} = (10^{{00}})$")
                fig.tight_layout()

                def update(frame):
                    frame = int(2*frame)
                    for line, x, y in plot_elements:
                        if x is not None:
                            line.set_xdata(x[frame])
                        if y is not None:
                            line.set_ydata(y[frame])

                    power = f" = {phB[mask][frame]:.3e}".split("e")
                    mass_name = f"{power[0]} \\times 10^{{{power[1]}}}"
                    for s,m in zip([general.MASS[k] for k in ("pi0", "eta", "etaP")], [solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME]):
                        if phB[mask][frame] == m:
                            power = f"{0:+.3e}".split("e")
                            mass_name = f"/{s} = (1{power[0]} \\times 10^{{{power[1]}}}) = {1:.5f}"
                            break
                        if 0.8 <= phB[mask][frame]/m <= 1.2:
                            power = f"{phB[mask][frame]/m-1:+.3e}".split("e")
                            mass_name = f"/{s} = (1{power[0]} \\times 10^{{{power[1]}}}) = {phB[mask][frame]/m:.5f}"
                            break
                    fig.suptitle(f"${general.MASS['phiBare']} {mass_name}$")
                    return [l for l,_,_ in plot_elements]

                match quarks:
                    case (1,1,1):
                        quark_name = "equal"
                    case (2,-1,-1):
                        quark_name = "charge"
                    case _:
                        quark_name = str(quarks)
                filename = f"{tolerance:.1e}_{quark_name}_{mask_name}"

                ani = animation.FuncAnimation(fig, update, frames=len(xs[mask]), interval=5000/len(xs[mask&~erratic]))
                ani.save(filename=f"animate_{filename}.gif", writer="pillow")
                ani.save(filename=f"animate_{filename}.mp4", writer="ffmpeg")

                plt.close()