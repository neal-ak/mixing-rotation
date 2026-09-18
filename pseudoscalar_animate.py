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

#     old_file = pathlib.Path() / "results_pseudo_tol8_hist.pkl"
#     if old_file.is_file():
#         with open(old_file, "rb") as f:
#             results = pickle.load(f)

#     for xs, saved, err, quarks, tol in results:
#         if tol not in stored:
#             stored[tol] = {}
#         if quarks not in stored[tol]:
#             stored[tol][quarks] = {}
#         if saved[general.MASS["phiBare"]][0] in stored[tol][quarks]:
#             raise ValueError("Bare Phi mass already present")
#         solved = {}
#         for k,v in saved.items():
#             if r"\tilde{g}" in str(k) or r"E_{" in str(k):
#                 break
#             solved[k] = v
#         m_phi_bare = solved[general.MASS["phiBare"]][0]
#         for k in [general.MASS["phiBare"], general.MASS["pi0"], general.MASS["eta"], general.MASS["etaP"], general.MASS["pi0-phi"], general.MASS["eta-phi"], general.MASS["etaP-phi"]]:
#             del solved[k]
#         stored[tol][quarks][saved[general.MASS["phiBare"]][0]] = {"coupling": xs}
#         stored[tol][quarks][saved[general.MASS["phiBare"]][0]] |= solved
#         finite = np.isfinite(stored[tol][quarks][saved[general.MASS["phiBare"]][0]][general.ANGLE["alpha"]])
#         for k,v in stored[tol][quarks][saved[general.MASS["phiBare"]][0]].items():
#             stored[tol][quarks][saved[general.MASS["phiBare"]][0]][k] = v[finite]

# # # %%
#     with open(stored_results, "wb") as f:
#         pickle.dump(stored, f)

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
    angle_style = dict(zip([general.ANGLE[k] for k in ["alpha", "beta", "gamma"]], tab20))
    mass_style = dict(zip([general.MASS[k] for k in ["phi", "pi0Bare", "etaBare", "etaPBare"]], [("black", "black")]+list(tab20)))
    meson_style = dict(zip([general.COUPLING[k] for k in ["phi-chi", "pi0-chi", "eta-chi", "etaP-chi", "phi-gamma"]], [("black", "black")]+list(tab20[:3])+[("purple", "purple")]))
    g_chi = 1
    for tolerance in stored:
        for quarks, scan in stored[tolerance].items():
            phB, xs = scan[general.MASS["phiBare"]], scan["coupling"]
            masks = {
                "pi": np.abs(phB/solver.MASS_NEUTRAL_PION - 1) < 0.11,
                "eta": np.abs(phB/solver.MASS_ETA - 1) < 0.11,
                "etaP": np.abs(phB/solver.MASS_ETA_PRIME - 1) < 0.11,
            }
            masks["other"] = ~masks["pi"] & ~masks["eta"] & ~masks["etaP"]
            m_diag = pseudo.pseudo_calculate_bare_masses(*(xs*np.array(quarks)[:,np.newaxis,np.newaxis]))
            approx = {general.ANGLE[a]: f(scan[general.MASS["phi"]], m_diag[general.MASS[m+"-phi"]]) for a, f, m in zip(("alpha", "beta", "gamma"), (small_alpha, small_beta, small_gamma), ("pi0", "eta", "etaP"))}
            approx |= {
                general.COUPLING["pi0-chi"]: approx[general.ANGLE["alpha"]] * g_chi,
                general.COUPLING["eta-chi"]: approx[general.ANGLE["beta"]] * g_chi,
                general.COUPLING["etaP-chi"]: approx[general.ANGLE["gamma"]] * g_chi,
                general.COUPLING["phi-gamma"]: -solver.G_PI_GAMMA*approx[general.ANGLE["alpha"]] - solver.G_ETA_GAMMA*approx[general.ANGLE["beta"]] - solver.G_ETA_PRIME_GAMMA*approx[general.ANGLE["gamma"]],
            }
            scan |= pseudo.pseudo_calculate_meson_couplings(4, g_chi, scan[general.ANGLE["alpha"]], scan[general.ANGLE["beta"]], scan[general.ANGLE["gamma"]])
            scan[general.COUPLING["phi-gamma"]] = pseudo.pseudo_calculate_photon_coupling(4, scan[general.ANGLE["alpha"]], scan[general.ANGLE["beta"]], scan[general.ANGLE["gamma"]], scan[general.ANGLE["delta"]], scan[general.ANGLE["zeta"]], scan[general.ANGLE["kappa"]])
            mask, mask_name = (), "all"
            for mask_name, mask in masks.items():
                fig, axs = plt.subplots(3,2, squeeze=False, figsize=(10,15), sharex=True)
                for label in angle_style:
                    axs[0,0].plot(xs[mask].T, scan[label][mask].T)
                    axs[0,0].plot(xs[mask].T, approx[label][mask].T)
                for label in mass_style:
                    axs[1,0].plot(xs[mask].T, scan[label][mask].T)
                for label in meson_style:
                    axs[2,0].plot(xs[mask].T, scan[label][mask].T)
                axs[0,0].set_xscale("log")
                axs[1,0].set_yscale("log")
                angle_limits = axs[0,0].get_xlim(), axs[0,0].get_ylim()
                mass_limits = axs[1,0].get_xlim(), axs[1,0].get_ylim()
                meson_limits = axs[2,0].get_xlim(), axs[2,0].get_ylim()
                plt.close()

                fig, axs = plt.subplots(3,2, squeeze=False, figsize=(10,15), sharex=True)
                for ax in axs[-1,:]:
                    ax.set_xlabel(f"$\\text{{Quark coupling strength: (}}{',\\quad'.join([f'{general.COUPLING[s]}={p}' for s,p in zip('uds',quarks)])} \\text{{)}}$")
                    ax.set_xlim(1e-7, 1e1)
                    ax.set_xscale("log")
                for ax in axs.flat:
                    ax.grid()

                # Angles
                angle_lines = []
                small_lines = []
                for label, colours in angle_style.items():
                    angle_lines.append(axs[0,0].plot(xs[mask][0], scan[label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.")[0])
                    small_lines.append(axs[0,0].plot(xs[mask][0], approx[label][mask][0], label=f"${label}_{{\\text{{approx}}}}$", color=colours[1], linewidth=1/2)[0])
                axs[0,0].set(
                    # ylim=(-np.pi/2, np.pi/2),
                    ylim=(-np.pi/16, np.pi/16),
                    ylabel="Angle [Rad]",
                )
                axs[0,0].legend(loc="lower left")
                if mask_name == "other":
                    axs[0,0].set_ylim(angle_limits[1])
                sec_y = axs[0,0].secondary_yaxis("right", functions=(lambda x: x*180/np.pi, lambda x: x*np.pi/180))
                sec_y.set_ylabel("Angle [Deg]")

                # Masses
                mass_lines = []
                phi_hline = axs[1,0].axhline(phB[0], color="black", linewidth=1/2)
                for label, colours in mass_style.items():
                    mass_lines.append(axs[1,0].plot(xs[mask][0], scan[label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.")[0])
                for m, colours in zip((solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME), tab20):
                    axs[1,0].axhline(m, color=colours[1], linewidth=1)
                axs[1,0].set(
                    ylim=mass_limits[1],
                    ylabel="Mass [GeV]",
                    yscale="log",
                )
                axs[1,0].legend(loc="lower left")

                # Couplings
                meson_lines = []
                axs[2,0].axhline(g_chi, color="black", linewidth=1/2, label=f"${general.COUPLING['chi']} = {g_chi}$")
                for label, colours in meson_style.items():
                    meson_lines.append(axs[2,0].plot(xs[mask][0], scan[label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.")[0])
                axs[2,0].set(
                    ylim=(-0.15,0.15),
                    ylabel="Coupling",
                )
                if mask_name == "other":
                    axs[2,0].set_ylim(meson_limits[1])
                axs[2,0].legend(loc="lower left")

                fig.suptitle(f"${general.MASS['phiBare']} = $")
                fig.tight_layout()

                def update(frame):
                    for line, small, label in zip(angle_lines, small_lines, angle_style):
                        line.set_data(xs[mask][frame], scan[label][mask][frame])
                        small.set_data(xs[mask][frame], approx[label][mask][frame])
                    for line, label in zip(mass_lines, mass_style):
                        line.set_data(xs[mask][frame], scan[label][mask][frame])
                    phi_hline.set_ydata([phB[mask][frame]])
                    for line, label in zip(meson_lines, meson_style):
                        line.set_data(xs[mask][frame], scan[label][mask][frame])

                    mass_name = f"{phB[mask][frame]:.3e}"
                    for s,m in zip([general.MASS[k] for k in ("pi0", "eta", "etaP")], [solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME]):
                        if phB[mask][frame] == m:
                            mass_name = f"{s}"
                            break
                        if 0.8 <= phB[mask][frame]/m <= 1.2:
                            mass_name = f"{phB[mask][frame]/m:.4f} {s}"
                            break
                    fig.suptitle(f"${general.MASS['phiBare']} = {mass_name}$")
                    return angle_lines+small_lines+mass_lines+[phi_hline]+meson_lines

                ani = animation.FuncAnimation(fig, update, frames=len(xs[mask]), interval=1000)
                match quarks:
                    case (1,1,1):
                        quark_name = "equal"
                    case (2,-1,-1):
                        quark_name = "charge"
                    case _:
                        quark_name = str(quarks)
                ani.save(filename=f"animate_{tolerance:.1e}_{quark_name}_{mask_name}.gif", writer="pillow")
                break
            break