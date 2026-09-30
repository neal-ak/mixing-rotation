# %%
import pathlib
import pickle

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from rotation.numerical import solver, pseudo
from rotation.symbolic import general
from rotation.symbolic.pseudo import bare_pseudo_masses

def small_alpha(m_phi, m_pi0_phi):
    return 0.5 * m_pi0_phi**2/(solver.MASS_NEUTRAL_PION**2 - m_phi**2)

def small_beta(m_phi, m_eta_phi):
    return 0.5 * m_eta_phi**2/(solver.MASS_ETA**2 - m_phi**2)

def small_gamma(m_phi, m_etaP_phi):
    return 0.5 * m_etaP_phi**2/(solver.MASS_ETA_PRIME**2 - m_phi**2)

def pseudo_mass_ratio():
    sx, sr = sym.symbols("x r")
    parameter = {
        general.MASS["u"]: solver.MASS_QUARK["up"],
        general.MASS["d"]: solver.MASS_QUARK["down"],
        general.MASS["s"]: solver.MASS_QUARK["strange"],
        general.MASS["pi0"]: solver.MASS_NEUTRAL_PION,
        general.MASS["eta"]: solver.MASS_ETA,
        general.MASS["etaP"]: solver.MASS_ETA_PRIME,
        general.PARAMETERS["B0"]: solver.B_0,
        general.PARAMETERS["F"]: solver.PION_DECAY,
        general.PARAMETERS["L6"]: solver.L_6,
        general.PARAMETERS["L7"]: solver.L_7,
        general.COUPLING["u"]: general.COUPLING["u"]*sx,
        general.COUPLING["d"]: general.COUPLING["d"]*sx,
        general.COUPLING["s"]: general.COUPLING["s"]*sx,
    }
    result = {}
    call = [general.COUPLING["u"], general.COUPLING["d"], general.COUPLING["s"], general.MASS["phiBare"]]
    bares = bare_pseudo_masses()
    for label, sol_num in zip(["pi0", "eta", "etaP"], [0,1,0]):
        ratio = bares[general.MASS[f"{label}-phi"]]**2 / sym.Abs(general.MASS[label]**2 - general.MASS["phiBare"]**2)
        ratio = ratio.subs(parameter)
        strength = sym.solve(sr - ratio, [sx], dict=True)[sol_num][sx]
        result[label] = (
            sym.lambdify([sx]+call, ratio),
            sym.lambdify([sr]+call, strength),
        )
    return result

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
    height, width, figsize = 2, 2, 5
    secondary_xs = pseudo_mass_ratio()
    for tolerance in [1e-8]:
        for quarks in [(2,-1,-1)]:
            phB, xs = stored[tolerance][quarks][general.MASS["phiBare"]], stored[tolerance][quarks]["coupling"]
            masks = {
                "other": (np.log10(phB)-np.log10(phB).min()) % np.diff(np.log10(phB))[0] < 1e-6,
                "pi0": np.abs(phB/solver.MASS_NEUTRAL_PION - 1) < 2e-1,
                "eta": np.abs(phB/solver.MASS_ETA - 1) < 2e-1,
                "etaP": np.abs(phB/solver.MASS_ETA_PRIME - 1) < 2e-1,
            }
            erratic_thresh = 1e-4
            erratic = (np.abs(phB/solver.MASS_NEUTRAL_PION-1)<=erratic_thresh) | (np.abs(phB/solver.MASS_ETA-1)<=erratic_thresh) | (np.abs(phB/solver.MASS_ETA_PRIME-1)<=erratic_thresh)

            for mask_name, mask in masks.items():
                fig, axs = plt.subplots(height,width, squeeze=False, figsize=figsize*np.array((width,height)))
                limits = np.empty((*axs.shape, 2))

                angle_ax = 0,0
                masses_ax = 0,1
                meson_ax = 1,0
                fractional_ax = 1,1
                # ratio_ax = 1,1

                # Prep limits
                try:
                # Angles
                    axs[angle_ax].set_xscale("log")
                    for label in angle_style:
                        axs[angle_ax].plot(xs[mask&~erratic].T, stored[tolerance][quarks][label][mask&~erratic].T)
                        if label in approx:
                            axs[angle_ax].plot(xs[mask&~erratic].T, approx[tolerance][quarks][label][mask&~erratic].T)
                    angle_limits = (
                        max(axs[angle_ax].get_ylim()[0], -1/2*np.pi),
                        min(axs[angle_ax].get_ylim()[1], 1/2*np.pi),
                    )
                except NameError:
                    pass

                try:
                    # Masses
                    axs[masses_ax].set_xscale("log")
                    for label in mass_style:
                        axs[masses_ax].plot(xs[mask&~erratic].T, stored[tolerance][quarks][label][mask&~erratic].T)
                        if label in approx:
                            axs[masses_ax].plot(xs[mask&~erratic].T, approx[tolerance][quarks][label][mask&~erratic].T)
                    mass_limits = (
                        max(axs[masses_ax].get_ylim()[0], 1e-3),
                        axs[masses_ax].get_ylim()[1],
                    )
                except NameError:
                    pass

                try:
                    # Meson couplings
                    axs[meson_ax].set_xscale("log")
                    for label in meson_style:
                        axs[meson_ax].plot(xs[mask&~erratic].T, stored[tolerance][quarks][label][mask&~erratic].T)
                        if label in approx:
                            axs[meson_ax].plot(xs[mask&~erratic].T, approx[tolerance][quarks][label][mask&~erratic].T)
                    meson_limits = (
                        max(axs[meson_ax].get_ylim()[0], -10*g_chi),
                        min(axs[meson_ax].get_ylim()[1], 10*g_chi),
                    )
                except NameError:
                    pass

                try:
                    # Ratio
                    axs[ratio_ax].set_xscale("log")
                    for label in angle_style|mass_style|meson_style:
                        if label in ratio[tolerance][quarks]:
                            axs[ratio_ax].plot(xs[mask&~erratic].T, ratio[tolerance][quarks][label][mask&~erratic].T)
                    ratio_limits = (
                        max(axs[ratio_ax].get_ylim()[0], -10),
                        min(axs[ratio_ax].get_ylim()[1], 10),
                    )
                except NameError:
                    pass

                try:
                    # Fractional error
                    axs[fractional_ax].set_xscale("log")
                    for label in angle_style|mass_style|meson_style:
                        if label in fraction[tolerance][quarks]:
                            axs[fractional_ax].plot(xs[mask&~erratic].T, fraction[tolerance][quarks][label][mask&~erratic].T)
                    _xs = np.logspace(-8,2,num=101)
                    _xs = np.tile(_xs, mask.shape[0]).reshape(mask.shape[0],-1)
                    fraction_limits = (
                        max(axs[fractional_ax].get_ylim()[0], 0),
                        min(axs[fractional_ax].get_ylim()[1], 1),
                    )
                except NameError:
                    pass
                plt.close()

                fig, axs = plt.subplots(height,width, squeeze=False, figsize=figsize*np.array((width,height)), sharex=True)
                for ax in axs[-1,:]:
                    ax.set_xlabel(f"$\\text{{Quark coupling strength: (}}{',\\quad'.join([f'{general.COUPLING[s]}={p}' for s,p in zip('uds',quarks)])} \\text{{)}}$")
                secondary_xaxes = []
                for index, ax in enumerate(axs.flat):
                    if mask_name == "other":
                        ax.set_xlim(1e-4, 1e1)
                    elif mask_name == "pi0":
                        ax.set_xlim(1e-5,1e-1)
                    elif mask_name == "eta":
                        ax.set_xlim(2e-5,2e-1)
                    elif mask_name == "etaP":
                        ax.set_xlim(3e-5,3e-1)
                    else:
                        ax.set_xlim(1e-7, 1e1)
                    ax.set_xscale("log")
                    if mask_name in secondary_xs:
                        secondary_xaxes.append(ax.secondary_xaxis(
                            "top",
                            functions=(
                                lambda x: secondary_xs[mask_name][0](x, *quarks, phB[mask][0]),
                                lambda x: secondary_xs[mask_name][1](x, *quarks, phB[mask][0]),
                            ),
                        ))
                        if index < width:
                            secondary_xaxes[-1].set_xlabel(f"${general.MASS[mask_name+'-phi']/sym.Abs(general.MASS[mask_name]-general.MASS['phiBare'])}$")
                        # else:
                        #     secondary_xaxes[-1].set_ticks([])

                plot_elements = []

                try:
                    # Angles
                    for label, colours in angle_style.items():
                        line = axs[angle_ax].plot(xs[mask][0], stored[tolerance][quarks][label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.", marker="")
                        plot_elements.append((line[0], xs[mask], stored[tolerance][quarks][label][mask]))
                        if label in approx[tolerance][quarks]:
                            line = axs[angle_ax].plot(xs[mask][0], approx[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(approx[tolerance][quarks][label][mask])) else ""+f"${label}_{{\\text{{approx}}}}$", color=colours[1], linewidth=1, marker="")
                            plot_elements.append((line[0], xs[mask], approx[tolerance][quarks][label][mask]))
                    axs[angle_ax].set(
                        ylim=angle_limits,
                        ylabel="Angle [Rad]",
                    )
                    axs[angle_ax].legend(loc="lower left")
                    sec_y = axs[angle_ax].secondary_yaxis("right", functions=(lambda x: x*180/np.pi, lambda x: x*np.pi/180))
                    sec_y.set_ylabel("Angle [Deg]")
                except NameError:
                    pass

                try:
                    # Masses
                    phi_hline = axs[masses_ax].axhline(phB[mask][0], color="black", linewidth=1)
                    plot_elements.append((phi_hline, None, phB[mask][...,np.newaxis]))
                    for label, colours in mass_style.items():
                        line = axs[masses_ax].plot(xs[mask][0], stored[tolerance][quarks][label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.", marker="")
                        plot_elements.append((line[0], xs[mask], stored[tolerance][quarks][label][mask]))
                    for m, colours in zip((solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME), tab20):
                        axs[masses_ax].axhline(m, color=colours[1], linewidth=1)
                    axs[masses_ax].set(
                        ylim=mass_limits,
                        ylabel="Mass [GeV]",
                        yscale="log",
                    )
                    axs[masses_ax].legend(loc="lower left")
                except NameError:
                    pass

                try:
                    # Meson Couplings
                    axs[meson_ax].axhline(g_chi, color=tab20[7,1], linewidth=1, label=f"${general.COUPLING['chi']} = {g_chi}$")
                    for label, colours in meson_style.items():
                        line = axs[meson_ax].plot(xs[mask][0], stored[tolerance][quarks][label][mask][0], label=f"${label}$", color=colours[0], linestyle="-.", marker="")
                        plot_elements.append((line[0], xs[mask], stored[tolerance][quarks][label][mask]))
                        if label in approx[tolerance][quarks]:
                            line = axs[meson_ax].plot(xs[mask][0], approx[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(approx[tolerance][quarks][label][mask])) else ""+f"${str(label)[:-1]},\\text{{approx}}}}$", color=colours[1], linewidth=1, marker="")
                            plot_elements.append((line[0], xs[mask], approx[tolerance][quarks][label][mask]))
                    axs[meson_ax].set(
                        ylim=meson_limits,
                        ylabel="Coupling",
                    )
                    axs[meson_ax].legend(loc="lower left")
                except NameError:
                    pass

                try:
                    # Ratio
                    for label, colours in angle_style.items():
                        if label in ratio[tolerance][quarks]:
                            line = axs[ratio_ax].plot(xs[mask][0], ratio[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(ratio[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle="-", linewidth=1, marker="")
                            plot_elements.append((line[0], xs[mask], ratio[tolerance][quarks][label][mask]))
                    for label, colours in mass_style.items():
                        if label in [general.MASS["phi"]]:
                            continue
                        if label in ratio[tolerance][quarks]:
                            line = axs[ratio_ax].plot(xs[mask][0], ratio[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(ratio[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle="-.", linewidth=1, marker="")
                            plot_elements.append((line[0], xs[mask], ratio[tolerance][quarks][label][mask]))
                    axs[ratio_ax].set(
                        ylim=ratio_limits,
                        ylabel="Approx / Reference",
                    )
                    axs[ratio_ax].legend(loc="lower left")
                except NameError:
                    pass

                try:
                    # Fractional error
                    for label, colours in mass_style.items():
                        if label not in [general.MASS["phi"]]:
                            continue
                        if label in fraction[tolerance][quarks]:
                            line = axs[fractional_ax].plot(xs[mask][0], fraction[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(fraction[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle="-.", linewidth=1, marker="")
                            plot_elements.append((line[0], xs[mask], fraction[tolerance][quarks][label][mask]))
                    for label, colours in meson_style.items():
                        if label in fraction[tolerance][quarks]:
                            line = axs[fractional_ax].plot(xs[mask][0], fraction[tolerance][quarks][label][mask][0], label="_" if np.all(np.isnan(fraction[tolerance][quarks][label][mask])) else ""+f"${label}$", color=colours[0], linestyle="-", linewidth=1, marker="")
                            plot_elements.append((line[0], xs[mask], fraction[tolerance][quarks][label][mask]))
                    axs[fractional_ax].set(
                        ylim=fraction_limits,
                        ylabel="| Approx / Reference - 1 |",
                    )
                    axs[fractional_ax].legend(loc="lower left")
                except NameError:
                    pass

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
                    if mask_name in secondary_xs and phB[mask][frame] not in [solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME]:
                        for index, ax in enumerate(axs.flat):
                            secondary_xaxes[index].remove()
                            secondary_xaxes[index] = ax.secondary_xaxis(
                                "top",
                                functions=(
                                    lambda x: secondary_xs[mask_name][0](x, *quarks, phB[mask][frame]),
                                    lambda x: secondary_xs[mask_name][1](x, *quarks, phB[mask][frame]),
                                ),
                            )
                            if index < width:
                                secondary_xaxes[index].set_xlabel(f"${general.MASS[mask_name+'-phi']/sym.Abs(general.MASS[mask_name]-general.MASS['phiBare'])}$")
                            # else:
                            #     secondary_xaxes[index].set_ticks([])
                    return [l for l,_,_ in plot_elements]

                match quarks:
                    case (1,1,1):
                        quark_name = "equal"
                    case (2,-1,-1):
                        quark_name = "charge"
                    case _:
                        quark_name = str(quarks)
                filename = f"{tolerance:.1e}_{quark_name}_{mask_name}"

                for d in [1-1e-5, 1-1e-1]:
                    for m in [solver.MASS_NEUTRAL_PION, solver.MASS_ETA, solver.MASS_ETA_PRIME]:
                        m = d*m
                        nearness = np.abs(phB[mask]/m-1)
                        target = np.argmin(nearness)
                        if nearness[target] > 1e-3:
                            continue
                        update(target)
                        fig.savefig(f"frame_{filename}_{target:0>4n}_{phB[mask][target]}.pdf")

                ani = animation.FuncAnimation(fig, update, frames=len(xs[mask]), interval=5000/len(xs[mask&~erratic]))
                ani.save(filename=f"animate_{filename}.gif", writer="pillow")
                ani.save(filename=f"animate_{filename}.mp4", writer="ffmpeg")

                plt.close()
