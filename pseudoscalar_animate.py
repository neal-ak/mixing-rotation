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
        for quarks, scan in stored[tolerance].items():
            phis = sorted(scan)
            stored[tolerance][quarks] = {k: v[np.newaxis, ...] for k,v in scan[phis[0]].items()}
            for p in phis[1:]:
                syn_shape = next(iter(stored[tolerance][quarks].values())).shape
                curr_shape = next(iter(scan[p].values())).shape
                if curr_shape[-1] > syn_shape[-1]:
                    for k,v in stored[tolerance][quarks].items():
                        stored[tolerance][quarks][k] = np.pad(v, ((0,0),(0,curr_shape[-1]-syn_shape[-1])), mode="constant", constant_values=np.nan)
                    syn_shape = *syn_shape[:-1], curr_shape[-1]
                for k,v in scan[p].items():
                    if k in stored[tolerance][quarks]:
                        if syn_shape[-1] > curr_shape[-1]:
                            v = np.pad(v, (0,syn_shape[-1]-curr_shape[-1]), mode="constant", constant_values=np.nan)
                        stored[tolerance][quarks][k] = np.append(stored[tolerance][quarks][k], v[np.newaxis, ...], axis=0)
                    else:
                        raise ValueError("Had an extra key not in others, they should all be the same.")
            stored[tolerance][quarks][general.MASS["phiBare"]] = np.array(phis)

# %%
    tab20 = np.array(plt.get_cmap("tab20").colors).reshape(10,2,3)
    angle_style = dict(zip([general.ANGLE[k] for k in ["alpha", "beta", "gamma", "delta", "zeta", "kappa"]], tab20))
    mass_style = dict(zip([general.MASS[k] for k in ["phi", "pi0Bare", "etaBare", "etaPBare"]], tab20[[7,0,1,2]]))
    meson_style = dict(zip([general.COUPLING[k] for k in ["phi-chi", "pi0-chi", "eta-chi", "etaP-chi", "phi-gamma"]], tab20[[7,0,1,2,6]]))
    g_chi = 1
    fractional = np.array((True, False, True), dtype=bool)
    height, width, figsize = 3, 2, 5
    for tolerance in [1e-8]:
        for quarks, scan in stored[tolerance].items():
            phB, xs = scan[general.MASS["phiBare"]], scan["coupling"]
            masks = {
                "pi": np.abs(phB/solver.MASS_NEUTRAL_PION - 1) < 2e-1,
                "eta": np.abs(phB/solver.MASS_ETA - 1) < 2e-1,
                "etaP": np.abs(phB/solver.MASS_ETA_PRIME - 1) < 2e-1,
            }
            masks["other"] = (np.log10(phB)-np.log10(phB).min()) % np.diff(np.log10(phB))[0] < 1e-3
            erratic = (np.abs(phB/solver.MASS_NEUTRAL_PION-1)<=1e-3) | (np.abs(phB/solver.MASS_ETA-1)<=1e-3) | (np.abs(phB/solver.MASS_ETA_PRIME-1)<=1e-3)

            scan |= pseudo.pseudo_calculate_meson_couplings(4, g_chi, scan[general.ANGLE["alpha"]], scan[general.ANGLE["beta"]], scan[general.ANGLE["gamma"]])
            scan[general.COUPLING["phi-gamma"]] = pseudo.pseudo_calculate_photon_coupling(4, scan[general.ANGLE["alpha"]], scan[general.ANGLE["beta"]], scan[general.ANGLE["gamma"]], scan[general.ANGLE["delta"]], scan[general.ANGLE["zeta"]], scan[general.ANGLE["kappa"]])
            m_diag = pseudo.pseudo_calculate_bare_masses(*(xs*np.array(quarks)[:,np.newaxis,np.newaxis]))
            approx = {general.ANGLE[a]: f(scan[general.MASS["phi"]], m_diag[general.MASS[m+"-phi"]]) for a, f, m in zip(("alpha", "beta", "gamma"), (small_alpha, small_beta, small_gamma), ("pi0", "eta", "etaP"))}
            approx |= {
                general.ANGLE["delta"]: np.nan * xs,
                general.ANGLE["zeta"]: np.nan * xs,
                general.ANGLE["kappa"]: np.nan * xs,
            }
            approx |= {
                general.COUPLING["phi-chi"]: np.nan * xs,
                general.COUPLING["pi0-chi"]: approx[general.ANGLE["alpha"]] * g_chi,
                general.COUPLING["eta-chi"]: approx[general.ANGLE["beta"]] * g_chi,
                general.COUPLING["etaP-chi"]: approx[general.ANGLE["gamma"]] * g_chi,
                general.COUPLING["phi-gamma"]: -solver.G_PI_GAMMA*approx[general.ANGLE["alpha"]] - solver.G_ETA_GAMMA*approx[general.ANGLE["beta"]] - solver.G_ETA_PRIME_GAMMA*approx[general.ANGLE["gamma"]],
            }
            ratio = {k: scan[k]/v for k,v in approx.items()}
            ratio |= {
                general.MASS["phi"]: scan[general.MASS["phi"]] / phB[...,np.newaxis],
                general.MASS["pi0Bare"]: solver.MASS_NEUTRAL_PION / scan[general.MASS["pi0Bare"]],
                general.MASS["etaBare"]: solver.MASS_ETA / scan[general.MASS["etaBare"]],
                general.MASS["etaPBare"]: solver.MASS_ETA_PRIME / scan[general.MASS["etaPBare"]],
            }
            if fractional[0]:
                for k in [_x for _x in general.ANGLE.values() if _x in ratio]:
                    ratio[k] = np.abs(ratio[k]-1)
            if fractional[1]:
                for k in [_x for _x in general.MASS.values() if _x in ratio]:
                    ratio[k] = np.abs(ratio[k]-1)
            if fractional[2]:
                for k in [_x for _x in general.COUPLING.values() if _x in ratio]:
                    ratio[k] = np.abs(ratio[k]-1)

            mask, mask_name = np.ones_like(next(iter(masks.values())), dtype=bool), "all"
            for mask_name, mask in masks.items():
                fig, axs = plt.subplots(height,width, squeeze=False, figsize=figsize*np.array((width,height)), sharex=True)
                limits = np.empty((*axs.shape, 2))
                axs[0,0].set_xscale("log")
                axs[1,0].set_yscale("log")
                for index, style in enumerate((angle_style, mass_style, meson_style)):
                    for label in style:
                        axs[index,0].plot(xs[mask&~erratic].T, scan[label][mask&~erratic].T, marker="")
                        if label in approx:
                            axs[index,0].plot(xs[mask&~erratic].T, approx[label][mask&~erratic].T, marker="")
                        if label in ratio:
                            axs[index,1].plot(xs[mask&~erratic].T, ratio[label][mask&~erratic].T, marker="")
                    axs[index,0].margins(0)
                    axs[index,1].margins(0)
                    limits[index,0,:] = axs[index,0].get_ylim()
                    limits[index,1,:] = axs[index,1].get_ylim()
                plt.close()
                limits[1,0][0] = max(limits[1,0][0], 1e-6)
                for index,l in enumerate(limits[:,1]):
                    limits[index,1] = (
                        max(l[0], -(1 if fractional[index] else 10)),
                        min(l[1], 1 if fractional[index] else 10),
                    )

                fig, axs = plt.subplots(height,width, squeeze=False, figsize=figsize*np.array((width,height)), sharex=True)
                for ax in axs[-1,:]:
                    ax.set_xlabel(f"$\\text{{Quark coupling strength: (}}{',\\quad'.join([f'{general.COUPLING[s]}={p}' for s,p in zip('uds',quarks)])} \\text{{)}}$")
                    ax.set_xlim(1e-7, 1e1)
                    ax.set_xscale("log")

                # Angles
                angle_lines, small_lines, angle_ratio = [], [], []
                for label, colours in angle_style.items():
                    angle_lines.append(axs[0,0].plot(xs[mask][0], scan[label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.", marker="")[0])
                    l = f"${label}_{{\\text{{approx}}}}$" if not np.all(np.isnan(approx[label][mask][0])) else ""
                    small_lines.append(axs[0,0].plot(xs[mask][0], approx[label][mask][0], label=l, color=colours[1], linewidth=1, marker="")[0])
                    l = f"${label}$" if not np.all(np.isnan(ratio[label][mask][0])) else ""
                    angle_ratio.append(axs[0,1].plot(xs[mask][0], ratio[label][mask][0], label=l, color=colours[0], marker="")[0])
                axs[0,0].set(
                    ylim=limits[0,0],
                    ylabel="Angle [Rad]",
                )
                axs[0,0].legend(loc="lower left")
                sec_y = axs[0,0].secondary_yaxis("right", functions=(lambda x: x*180/np.pi, lambda x: x*np.pi/180))
                sec_y.set_ylabel("Angle [Deg]")
                axs[0,1].set(
                    ylim=limits[0,1],
                    ylabel=("|" if fractional[0] else "")+"Angle / Small Angle Approximation"+(" - 1|" if fractional[0] else ""),
                )
                axs[0,1].legend(loc="lower left")

                # Masses
                mass_lines, mass_ratio = [], []
                phi_hline = axs[1,0].axhline(phB[0], color="black", linewidth=1)
                for label, colours in mass_style.items():
                    mass_lines.append(axs[1,0].plot(xs[mask][0], scan[label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.", marker="")[0])
                    l = f"${label}$" if not np.all(np.isnan(ratio[label][mask][0])) else ""
                    mass_ratio.append(axs[1,1].plot(xs[mask][0], ratio[label][mask][0], label=l, color=colours[0], marker="")[0])
                for m, colours in zip((solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME), tab20):
                    axs[1,0].axhline(m, color=colours[1], linewidth=1)
                axs[1,0].set(
                    ylim=(max(limits[1,0][0], 1e-6), limits[1,0][1]),
                    ylabel="Mass [GeV]",
                    yscale="log",
                )
                axs[1,0].legend(loc="lower left")
                axs[1,1].set(
                    ylim=limits[1,1],
                    ylabel=("|" if fractional[1] else "")+"Mass / Bare Mass"+(" - 1|" if fractional[1] else ""),
                )
                axs[1,1].legend(loc="lower left")

                # Couplings
                meson_lines, approx_lines, meson_ratio = [], [], []
                axs[2,0].axhline(g_chi, color=tab20[7,1], linewidth=1, label=f"${general.COUPLING['chi']} = {g_chi}$")
                for label, colours in meson_style.items():
                    meson_lines.append(axs[2,0].plot(xs[mask][0], scan[label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.", marker="")[0])
                    l = f"${str(label)[:-1]},\\text{{approx}}}}$" if not np.all(np.isnan(approx[label][mask][0])) else ""
                    approx_lines.append(axs[2,0].plot(xs[mask][0], approx[label][mask][0], label=l, color=colours[1], linewidth=1, marker="")[0])
                    l = f"${label}$" if not np.all(np.isnan(ratio[label][mask][0])) else ""
                    meson_ratio.append(axs[2,1].plot(xs[mask][0], ratio[label][mask][0], label=l, color=colours[0], marker="")[0])
                axs[2,0].set(
                    ylim=limits[2,0],
                    ylabel="Coupling",
                )
                axs[2,0].legend(loc="lower left")
                axs[2,1].set(
                    ylim=limits[2,1],
                    ylabel=("|" if fractional[2] else "")+"Coupling / Coupling Approximation"+(" - 1|" if fractional[2] else ""),
                )
                axs[2,1].legend(loc="lower left")

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

                fig.suptitle(f"${general.MASS['phiBare']} = $")
                fig.tight_layout()

                def update(frame):
                    for line, small, rat, label in zip(angle_lines, small_lines, angle_ratio, angle_style):
                        line.set_data(xs[mask][frame], scan[label][mask][frame])
                        small.set_data(xs[mask][frame], approx[label][mask][frame])
                        rat.set_data(xs[mask][frame], ratio[label][mask][frame])
                    for line, rat, label in zip(mass_lines, mass_ratio, mass_style):
                        line.set_data(xs[mask][frame], scan[label][mask][frame])
                        rat.set_data(xs[mask][frame], ratio[label][mask][frame])
                    phi_hline.set_ydata([phB[mask][frame]])
                    for line, app, rat, label in zip(meson_lines, approx_lines, meson_ratio, meson_style):
                        line.set_data(xs[mask][frame], scan[label][mask][frame])
                        app.set_data(xs[mask][frame], approx[label][mask][frame])
                        rat.set_data(xs[mask][frame], ratio[label][mask][frame])

                    mass_name = f"{phB[mask][frame]:.3e}"
                    for s,m in zip([general.MASS[k] for k in ("pi0", "eta", "etaP")], [solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME]):
                        if phB[mask][frame] == m:
                            mass_name = f"{1.0:.5f} {s}"
                            break
                        if 0.8 <= phB[mask][frame]/m <= 1.2:
                            mass_name = f"{phB[mask][frame]/m:.5f} {s}"
                            break
                    fig.suptitle(f"${general.MASS['phiBare']} = {mass_name}$")
                    return angle_lines+small_lines+mass_lines+[phi_hline]+meson_lines+approx_lines

                ani = animation.FuncAnimation(fig, update, frames=len(xs[mask]), interval=3000/len(xs[mask&~erratic]))
                match quarks:
                    case (1,1,1):
                        quark_name = "equal"
                    case (2,-1,-1):
                        quark_name = "charge"
                    case _:
                        quark_name = str(quarks)
                ani.save(filename=f"animate_{tolerance:.1e}_{quark_name}_{mask_name}.gif", writer="pillow")
                plt.close()