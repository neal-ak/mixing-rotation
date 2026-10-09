# %%
import pathlib
import pickle

from matplotlib import animation, rcParams
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from rotation.numerical import solver, pseudo
from rotation.symbolic import general
from rotation.symbolic.pseudo import bare_pseudo_masses

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_SCALE = 1.0
rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, # cmr10 has no Unicode minus
    "font.size": 17 * FONT_SCALE,
    "axes.labelsize": 24 * FONT_SCALE,
    "axes.titlesize": 24 * FONT_SCALE,
    "xtick.labelsize": 19 * FONT_SCALE,
    "ytick.labelsize": 19 * FONT_SCALE,
    "legend.fontsize": 19 * FONT_SCALE,
})


def small_alpha(m_phi, m_pi0_phi):
    return 0.5 * m_pi0_phi**2 / (solver.MASS_NEUTRAL_PION**2 - m_phi**2)


def small_beta(m_phi, m_eta_phi):
    return 0.5 * m_eta_phi**2 / (solver.MASS_ETA**2 - m_phi**2)


def small_gamma(m_phi, m_etaP_phi):
    return 0.5 * m_etaP_phi**2 / (solver.MASS_ETA_PRIME**2 - m_phi**2)


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
        general.COUPLING["u"]: general.COUPLING["u"] * sx,
        general.COUPLING["d"]: general.COUPLING["d"] * sx,
        general.COUPLING["s"]: general.COUPLING["s"] * sx,
    }
    result = {}
    call = [
        general.COUPLING["u"],
        general.COUPLING["d"],
        general.COUPLING["s"],
        general.MASS["phiBare"],
    ]
    bares = bare_pseudo_masses()
    for label, sol_num in zip(["pi0", "eta", "etaP"], [0, 1, 0]):
        ratio = bares[general.MASS[f"{label}-phi"]] ** 2 / sym.Abs(
            general.MASS[label] ** 2 - general.MASS["phiBare"] ** 2
        )
        ratio = ratio.subs(parameter)
        strength = sym.solve(sr - ratio, [sx], dict=True)[sol_num][sx]
        result[label] = (
            sym.lambdify([sx] + call, ratio),
            sym.lambdify([sr] + call, strength),
        )
    return result


def draw_angles(
    ax,
    x_strength: np.NDArray,
    alpha: tuple[np.NDArray, dict] = None,
    beta: tuple[np.NDArray, dict] = None,
    gamma: tuple[np.NDArray, dict] = None,
    delta: tuple[np.NDArray, dict] = None,
    zeta: tuple[np.NDArray, dict] = None,
    kappa: tuple[np.NDArray, dict] = None,
    alpha_approx: tuple[np.NDArray, dict] = None,
    beta_approx: tuple[np.NDArray, dict] = None,
    gamma_approx: tuple[np.NDArray, dict] = None,
):
    draw_order = tuple(
        _x
        for _x in [
            alpha,
            alpha_approx,
            beta,
            beta_approx,
            gamma,
            gamma_approx,
            delta,
            zeta,
            kappa,
        ]
        if _x is not None
    )
    limits = [np.nanmin(draw_order[0][0]), np.nanmax(draw_order[0][0])]
    for ys, _ in draw_order[1:]:
        limits[0] = min(limits[0], np.nanmin(ys))
        limits[1] = max(limits[1], np.nanmax(ys))
    limits = (
        max(limits[0], -1 / 2 * np.pi),
        min(limits[1], 1 / 2 * np.pi),
    )

    elements = []
    for ys, style in draw_order:
        if np.all(np.abs(ys[~np.isnan(ys)]) < 1e-14):
            continue  # if it's exactly zero, we don't need to show it
        elements.append((ax.plot(x_strength[0], ys[0], **style)[0], x_strength, ys))
    ax.set(
        xscale="log",
        ylim=limits,
        ylabel="Angle [Rad]",
    )
    ax.legend(loc="lower left")
    sec_y = ax.secondary_yaxis(
        "right", functions=(lambda x: x * 180 / np.pi, lambda x: x * np.pi / 180)
    )
    sec_y.set_ylabel("Angle [Deg]")
    return elements


def draw_masses(
    ax,
    x_strength: np.NDArray,
    m_phi: tuple[np.NDArray, dict] = None,
    m_pi0Bare: tuple[np.NDArray, dict] = None,
    m_etaBare: tuple[np.NDArray, dict] = None,
    m_etaPBare: tuple[np.NDArray, dict] = None,
    m_phiBare: tuple[np.NDArray, dict] = None,
    m_pi0: tuple[np.NDArray, dict] = None,
    m_eta: tuple[np.NDArray, dict] = None,
    m_etaP: tuple[np.NDArray, dict] = None,
):
    draw_order = tuple(
        _x for _x in [m_phi, m_pi0Bare, m_etaBare, m_etaPBare] if _x is not None
    )
    limits = [np.nanmin(draw_order[0][0]), np.nanmax(draw_order[0][0])]
    for ys, _ in draw_order[1:]:
        limits[0] = min(limits[0], np.nanmin(ys))
        limits[1] = max(limits[1], np.nanmax(ys))
    limits = (
        max(limits[0], 1e-3),
        limits[1],
    )

    elements = []
    if m_phiBare is not None:
        elements.append(
            (
                ax.axhline(m_phiBare[0][0], **m_phiBare[1]),
                None,
                m_phiBare[0][..., np.newaxis],
            )
        )
    for ys, style in draw_order:
        elements.append((ax.plot(x_strength[0], ys[0], **style)[0], x_strength, ys))
    if m_pi0 is None:
        m_pi0 = solver.MASS_NEUTRAL_PION, {}
    if m_eta is None:
        m_eta = solver.MASS_ETA, {}
    if m_etaP is None:
        m_etaP = solver.MASS_ETA_PRIME, {}
    for ys, style in [m_pi0, m_eta, m_etaP]:
        ax.axhline(ys, **style)
    ax.set(
        yscale="log",
        ylim=limits,
        ylabel="Mass [GeV]",
    )
    ax.legend(loc="lower left")
    return elements


def draw_mesons(
    ax,
    x_strength: np.NDArray,
    g_chi: tuple[float, dict],
    phi_chi: tuple[np.NDArray, dict] = None,
    pi0_chi: tuple[np.NDArray, dict] = None,
    eta_chi: tuple[np.NDArray, dict] = None,
    etaP_chi: tuple[np.NDArray, dict] = None,
    gamma_chi: tuple[np.NDArray, dict] = None,
    phi_gamma: tuple[np.NDArray, dict] = None,
    pi0_chi_approx: tuple[np.NDArray, dict] = None,
    eta_chi_approx: tuple[np.NDArray, dict] = None,
    etaP_chi_approx: tuple[np.NDArray, dict] = None,
    gamma_chi_approx: tuple[np.NDArray, dict] = None,
    phi_gamma_approx: tuple[np.NDArray, dict] = None,
):
    draw_order = tuple(
        _x
        for _x in [
            phi_chi,
            pi0_chi,
            eta_chi,
            etaP_chi,
            phi_gamma,
            pi0_chi_approx,
            eta_chi_approx,
            etaP_chi_approx,
            phi_gamma_approx,
        ]
        if _x is not None
    )
    limits = [np.nanmin(draw_order[0][0]), np.nanmax(draw_order[0][0])]
    for ys, _ in draw_order[1:]:
        limits[0] = min(limits[0], np.nanmin(ys))
        limits[1] = max(limits[1], np.nanmax(ys))
    limits = (
        max(limits[0], -g_chi[0]),
        min(limits[1], g_chi[0]),
    )

    ax.axhline(g_chi[0], **g_chi[1])
    elements = []
    for ys, style in draw_order:
        if np.all(np.abs(ys[~np.isnan(ys)]) < 1e-14):
            continue  # if it's exactly zero, we don't need to show it
        elements.append((ax.plot(x_strength[0], ys[0], **style)[0], x_strength, ys))
    ax.set(
        ylim=limits,
        ylabel="Coupling",
    )
    ax.legend(loc="lower left")
    return elements


def draw_ratios(
    ax,
    x_strength: np.NDArray,
    alpha: tuple[np.NDArray, dict] = None,
    beta: tuple[np.NDArray, dict] = None,
    gamma: tuple[np.NDArray, dict] = None,
    m_pi0Bare: tuple[np.NDArray, dict] = None,
    m_etaBare: tuple[np.NDArray, dict] = None,
    m_etaPBare: tuple[np.NDArray, dict] = None,
):
    draw_order = tuple(
        _x
        for _x in [alpha, beta, gamma, m_pi0Bare, m_etaBare, m_etaPBare]
        if _x is not None
    )
    limits = [np.nanmin(draw_order[0][0]), np.nanmax(draw_order[0][0])]
    for ys, _ in draw_order[1:]:
        limits[0] = min(limits[0], np.nanmin(ys))
        limits[1] = max(limits[1], np.nanmax(ys))
    limits = (
        max(limits[0], 10),
        min(limits[1], -10),
    )

    elements = []
    for ys, style in draw_order:
        if np.all(np.abs(ys[~np.isnan(ys)]) < 1e-14):
            continue  # if it's exactly zero, we don't need to show it
        elements.append((ax.plot(x_strength[0], ys[0], **style)[0], x_strength, ys))
    ax.set(
        ylim=limits[::-1],
        ylabel="Approx / Reference",
    )
    ax.legend(loc="lower left")
    return elements


def draw_fractions(
    ax,
    x_strength: np.NDArray,
    m_phi: tuple[np.NDArray, dict] = None,
    pi0_chi: tuple[np.NDArray, dict] = None,
    eta_chi: tuple[np.NDArray, dict] = None,
    etaP_chi: tuple[np.NDArray, dict] = None,
    phi_gamma: tuple[np.NDArray, dict] = None,
):
    draw_order = tuple(
        _x for _x in [m_phi, pi0_chi, eta_chi, etaP_chi, phi_gamma] if _x is not None
    )
    limits = [np.nanmin(draw_order[0][0]), np.nanmax(draw_order[0][0])]
    for ys, _ in draw_order[1:]:
        limits[0] = min(limits[0], np.nanmin(ys))
        limits[1] = max(limits[1], np.nanmax(ys))
    limits = (
        max(limits[0], 0),
        min(limits[1], 1),
    )

    elements = []
    for ys, style in draw_order:
        if np.all(np.abs(ys[~np.isnan(ys)] - 1) < 1e-14):
            continue  # if it's exactly zero, we don't need to show it
        elements.append((ax.plot(x_strength[0], ys[0], **style)[0], x_strength, ys))
    ax.set(
        yscale="log",
        ylim=limits,
        ylabel="Abs(Approx / Reference - 1)",
    )
    ax.legend(loc="lower left")
    return elements


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
            stored[tolerance][quarks] = {
                k: v[np.newaxis, ...] for k, v in scan[phis[0]].items()
            }
            for p in phis[1:]:
                syn_shape = next(iter(stored[tolerance][quarks].values())).shape
                curr_shape = next(iter(scan[p].values())).shape
                if curr_shape[-1] > syn_shape[-1]:
                    for k, v in stored[tolerance][quarks].items():
                        stored[tolerance][quarks][k] = np.pad(
                            v,
                            ((0, 0), (0, curr_shape[-1] - syn_shape[-1])),
                            mode="constant",
                            constant_values=np.nan,
                        )
                    syn_shape = *syn_shape[:-1], curr_shape[-1]
                for k, v in scan[p].items():
                    if k in stored[tolerance][quarks]:
                        if syn_shape[-1] > curr_shape[-1]:
                            v = np.pad(
                                v,
                                (0, syn_shape[-1] - curr_shape[-1]),
                                mode="constant",
                                constant_values=np.nan,
                            )
                        stored[tolerance][quarks][k] = np.append(
                            stored[tolerance][quarks][k], v[np.newaxis, ...], axis=0
                        )
                    else:
                        raise ValueError(
                            "Had an extra key not in others, they should all be the same."
                        )
            stored[tolerance][quarks][general.MASS["phiBare"]] = np.array(phis)

    g_chi = 1
    approx, ratio, fraction = {}, {}, {}
    for tolerance in stored:
        approx[tolerance], ratio[tolerance], fraction[tolerance] = {}, {}, {}
        for quarks in stored[tolerance]:
            stored[tolerance][quarks] |= pseudo.pseudo_calculate_meson_couplings(
                n_dim=4,
                bare_mediator_dm_coupling=g_chi,
                alpha=stored[tolerance][quarks][general.ANGLE["alpha"]],
                beta=stored[tolerance][quarks][general.ANGLE["beta"]],
                gamma=stored[tolerance][quarks][general.ANGLE["gamma"]],
            )
            stored[tolerance][quarks][general.COUPLING["phi-gamma"]] = (
                pseudo.pseudo_calculate_photon_coupling(
                    n_dim=4,
                    alpha=stored[tolerance][quarks][general.ANGLE["alpha"]],
                    beta=stored[tolerance][quarks][general.ANGLE["beta"]],
                    gamma=stored[tolerance][quarks][general.ANGLE["gamma"]],
                    delta=stored[tolerance][quarks][general.ANGLE["delta"]],
                    zeta=stored[tolerance][quarks][general.ANGLE["zeta"]],
                    kappa=stored[tolerance][quarks][general.ANGLE["kappa"]],
                )
            )
            m_diag = pseudo.pseudo_calculate_bare_masses(
                *(
                    stored[tolerance][quarks]["coupling"]
                    * np.array(quarks)[:, np.newaxis, np.newaxis]
                )
            )
            approx[tolerance][quarks] = {
                general.ANGLE[a]: f(
                    stored[tolerance][quarks][general.MASS["phi"]],
                    m_diag[general.MASS[m + "-phi"]],
                )
                for a, f, m in zip(
                    ("alpha", "beta", "gamma"),
                    (small_alpha, small_beta, small_gamma),
                    ("pi0", "eta", "etaP"),
                )
            }
            approx[tolerance][quarks] |= {
                general.COUPLING["pi0-chi"]: approx[tolerance][quarks][
                    general.ANGLE["alpha"]
                ]
                * g_chi,
                general.COUPLING["eta-chi"]: approx[tolerance][quarks][
                    general.ANGLE["beta"]
                ]
                * g_chi,
                general.COUPLING["etaP-chi"]: approx[tolerance][quarks][
                    general.ANGLE["gamma"]
                ]
                * g_chi,
                general.COUPLING["phi-gamma"]: -solver.G_PI_GAMMA
                * approx[tolerance][quarks][general.ANGLE["alpha"]]
                - solver.G_ETA_GAMMA * approx[tolerance][quarks][general.ANGLE["beta"]]
                - solver.G_ETA_PRIME_GAMMA
                * approx[tolerance][quarks][general.ANGLE["gamma"]],
            }
            ratio[tolerance][quarks] = {
                k: v / stored[tolerance][quarks][k]
                for k, v in approx[tolerance][quarks].items()
            }
            ratio[tolerance][quarks] |= {
                general.MASS["phi"]: stored[tolerance][quarks][general.MASS["phi"]]
                / stored[tolerance][quarks][general.MASS["phiBare"]][..., np.newaxis],
                general.MASS["pi0Bare"]: solver.MASS_NEUTRAL_PION
                / stored[tolerance][quarks][general.MASS["pi0Bare"]],
                general.MASS["etaBare"]: solver.MASS_ETA
                / stored[tolerance][quarks][general.MASS["etaBare"]],
                general.MASS["etaPBare"]: solver.MASS_ETA_PRIME
                / stored[tolerance][quarks][general.MASS["etaPBare"]],
            }
            fraction[tolerance][quarks] = {
                k: np.abs(v - 1) for k, v in ratio[tolerance][quarks].items()
            }

    # %%
    tab20 = np.array(plt.get_cmap("tab20").colors).reshape(10, 2, 3)
    angle_style = dict(
        zip(
            [
                general.ANGLE[k]
                for k in [
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "zeta",
                    "kappa",
                ]
            ],
            tab20,
        )
    )
    mass_style = dict(
        zip(
            [general.MASS[k] for k in ["phi", "pi0Bare", "etaBare", "etaPBare"]],
            tab20[[7, 0, 1, 2]],
        )
    )
    meson_style = dict(
        zip(
            [
                general.COUPLING[k]
                for k in ["phi-chi", "pi0-chi", "eta-chi", "etaP-chi", "phi-gamma"]
            ],
            tab20[[7, 0, 1, 2, 6]],
        )
    )
    g_chi = 1
    fractional = np.array((True, False, True), dtype=bool)
    height, width, figsize = 2, 2, np.array((7.8, 5.8))
    secondary_xs = pseudo_mass_ratio()
    for tolerance in [1e-8]:
        for quarks in [(2, -1, -1)]:
            phB, xs = (
                stored[tolerance][quarks][general.MASS["phiBare"]],
                stored[tolerance][quarks]["coupling"],
            )
            masks = {
                "other": (np.log10(phB) - np.log10(phB).min())
                % np.diff(np.log10(phB))[0]
                < 1e-6,
                "pi0": np.abs(phB / solver.MASS_NEUTRAL_PION - 1) < 2e-1,
                "eta": np.abs(phB / solver.MASS_ETA - 1) < 2e-1,
                "etaP": np.abs(phB / solver.MASS_ETA_PRIME - 1) < 2e-1,
            }

            for mask_name, mask in masks.items():
                fig, axs = plt.subplots(
                    height,
                    width,
                    squeeze=False,
                    figsize=figsize * np.array((width, height)),
                )
                fig_frame, ax_frame = plt.subplots(
                    1, 1, figsize=figsize
                )
                limits = np.empty((*axs.shape, 2))

                plot_elements = []
                plot_elements += draw_angles(
                    axs[0, 0],
                    xs[mask],
                    **{
                        k: (
                            stored[tolerance][quarks][general.ANGLE[k]][mask],
                            {
                                "label": f"${general.ANGLE[k]}$",
                                "color": angle_style[general.ANGLE[k]][0],
                                "linestyle": "-.",
                            },
                        )
                        for k in general.ANGLE
                        if general.ANGLE[k] in stored[tolerance][quarks]
                    }
                    | {
                        k
                        + "_approx": (
                            approx[tolerance][quarks][general.ANGLE[k]][mask],
                            {
                                "label": f"${general.ANGLE[k]}_\\text{{approx}}$",
                                "color": angle_style[general.ANGLE[k]][1],
                                "linewidth": 1,
                            },
                        )
                        for k in general.ANGLE
                        if general.ANGLE[k] in approx[tolerance][quarks]
                    },
                )

                plot_elements += draw_masses(
                    axs[0, 1],
                    xs[mask],
                    m_phiBare=(phB[mask], {"color": tab20[7, 1], "linewidth": 1}),
                    **{
                        "m_"
                        + k: (
                            stored[tolerance][quarks][general.MASS[k]][mask],
                            {
                                "label": f"${general.MASS[k]}$",
                                "color": mass_style[general.MASS[k]][0],
                                "linestyle": "-.",
                            },
                        )
                        for k in ["phi", "pi0Bare", "etaBare", "etaPBare"]
                    }
                    | {
                        "m_"
                        + k: (
                            m,
                            {
                                "label": f"${general.MASS[k]}$",
                                "color": mass_style[general.MASS[k + "Bare"]][1],
                                "linewidth": 1,
                            },
                        )
                        for k, m in zip(
                            ("pi0", "eta", "etaP"),
                            (
                                solver.MASS_NEUTRAL_PION,
                                solver.MASS_ETA,
                                solver.MASS_ETA_PRIME,
                            ),
                        )
                    },
                )

                plot_elements += draw_mesons(
                    axs[1, 0],
                    xs[mask],
                    g_chi=(
                        g_chi,
                        {
                            "label": f"${general.COUPLING['chi']} = {g_chi}$",
                            "color": tab20[7, 1],
                            "linewidth": 1,
                        },
                    ),
                    **{
                        k.replace("-", "_"): (
                            stored[tolerance][quarks][general.COUPLING[k]][mask],
                            {
                                "label": f"${general.COUPLING[k]}$",
                                "color": meson_style[general.COUPLING[k]][0],
                                "linestyle": "-.",
                            },
                        )
                        for k in [
                            "phi-chi",
                            "pi0-chi",
                            "eta-chi",
                            "etaP-chi",
                            "phi-gamma",
                        ]
                    }
                    | {
                        k.replace("-", "_")
                        + "_approx": (
                            approx[tolerance][quarks][general.COUPLING[k]][mask],
                            {
                                "label": f"${str(general.COUPLING[k])[:-1]}, approx}}$",
                                "color": meson_style[general.COUPLING[k]][1],
                            },
                        )
                        for k in ["pi0-chi", "eta-chi", "etaP-chi", "phi-gamma"]
                    },
                )

                plot_elements += draw_ratios(
                    axs[1, 1],
                    xs[mask],
                    **{
                        k: (
                            ratio[tolerance][quarks][general.ANGLE[k]][mask],
                            {
                                "label": f"${general.ANGLE[k]}$",
                                "color": angle_style[general.ANGLE[k]][0],
                            },
                        )
                        for k in ["alpha", "beta", "gamma"]
                    }
                    | {
                        "m_"
                        + k: (
                            ratio[tolerance][quarks][general.MASS[k]][mask],
                            {
                                "label": f"${general.MASS[k]}$",
                                "color": mass_style[general.MASS[k]][0],
                                "linestyle": "-.",
                            },
                        )
                        for k in ["pi0Bare", "etaBare", "etaPBare"]
                    },
                )

                for ax in [ax_frame]:
                    plot_elements += draw_fractions(
                        ax,
                        xs[mask],
                        m_phi=(
                            fraction[tolerance][quarks][general.MASS["phi"]][mask],
                            {
                                "label": f"${general.MASS['phi']}$",
                                "color": mass_style[general.MASS["phi"]][0],
                                "linestyle": "-.",
                            },
                        ),
                        **{
                            k.replace("-", "_"): (
                                fraction[tolerance][quarks][general.COUPLING[k]][mask],
                                {
                                    "label": f"${general.COUPLING[k]}$",
                                    "color": meson_style[general.COUPLING[k]][0],
                                },
                            )
                            for k in ["pi0-chi", "eta-chi", "etaP-chi", "phi-gamma"]
                        },
                    )

                for ax in list(axs[-1, :]) + [ax_frame]:
                    ax.set_xlabel("Q")
                secondary_xaxes = []
                match mask_name:
                    case "other":
                        xlims = (1e-4, 1e1)
                    case "pi0":
                        xlims = (1e-5, 1e-1)
                    case "eta":
                        xlims = (2e-5, 2e-1)
                    case "etaP":
                        xlims = (3e-5, 3e-1)
                    case _:
                        xlims = (1e-7, 1e1)
                for index, ax in enumerate(list(axs.flat) + [ax_frame]):
                    ax.set_xlim(*xlims)
                    ax.grid()
                    ax.set_xscale("log")
                    if mask_name in secondary_xs:
                        secondary_xaxes.append(
                            ax.secondary_xaxis(
                                "top",
                                functions=(
                                    lambda x: secondary_xs[mask_name][0](
                                        x, *quarks, phB[mask][0]
                                    ),
                                    lambda x: secondary_xs[mask_name][1](
                                        x, *quarks, phB[mask][0]
                                    ),
                                ),
                            )
                        )
                        secondary_xaxes[-1].set_xticks(np.logspace(-20,20,num=41))
                        if index < width or index >= height * width:
                            secondary_xaxes[-1].set_xlabel(
                                f"${general.MASS[mask_name+'-phi']/sym.Abs(general.MASS[mask_name]-general.MASS['phiBare'])}$"
                            )
                    if ax.get_yscale() == "log":
                        ax.set_ylim(
                            10
                            ** (
                                np.log10(ax.get_ylim())[0]
                                - 0.05 * np.diff(np.log10(ax.get_ylim()))[0]
                            ),
                            10
                            ** (
                                np.log10(ax.get_ylim())[1]
                                + 0.05 * np.diff(np.log10(ax.get_ylim()))[0]
                            ),
                        )
                    elif ax.get_yscale() == "linear":
                        ax.set_ylim(
                            ax.get_ylim()[0] - 0.05 * np.diff(ax.get_ylim())[0],
                            ax.get_ylim()[1] + 0.05 * np.diff(ax.get_ylim())[0],
                        )

                fig.suptitle(f"${general.MASS['phiBare']} = (10^{{00}})$")
                fig.tight_layout()

                fig_frame.suptitle(f"${general.MASS['phiBare']} = (10^{{00}})$")
                fig_frame.tight_layout()

                def update(frame):
                    for line, x, y in plot_elements:
                        if x is not None:
                            line.set_xdata(x[frame])
                        if y is not None:
                            line.set_ydata(y[frame])

                    power = f" = {phB[mask][frame]:.3e}".split("e")
                    mass_name = f"{power[0]} \\times 10^{{{power[1]}}}"
                    for s, m in zip(
                        [general.MASS[k] for k in ("pi0", "eta", "etaP")],
                        [
                            solver.MASS_NEUTRAL_PION,
                            solver.MASS_ETA,
                            solver.MASS_ETA_PRIME,
                        ],
                    ):
                        if phB[mask][frame] == m:
                            power = f"{0:+.3e}".split("e")
                            mass_name = f"/{s} = (1{power[0]} \\times 10^{{{power[1]}}}) = {1:.5f}"
                            break
                        if 0.8 <= phB[mask][frame] / m <= 1.2:
                            power = f"{phB[mask][frame]/m-1:+.3e}".split("e")
                            mass_name = f"/{s} = (1{power[0]} \\times 10^{{{power[1]}}}) = {phB[mask][frame]/m:.5f}"
                            break
                    fig.suptitle(f"${general.MASS['phiBare']} {mass_name}$")
                    fig_frame.suptitle(f"${general.MASS['phiBare']} {mass_name}$")
                    if mask_name in secondary_xs and phB[mask][frame] not in [
                        solver.MASS_NEUTRAL_PION,
                        solver.MASS_ETA,
                        solver.MASS_ETA_PRIME,
                    ]:
                        for index, ax in enumerate(list(axs.flat) + [ax_frame]):
                            secondary_xaxes[index].remove()
                            secondary_xaxes[index] = ax.secondary_xaxis(
                                "top",
                                functions=(
                                    lambda x: secondary_xs[mask_name][0](
                                        x, *quarks, phB[mask][frame]
                                    ),
                                    lambda x: secondary_xs[mask_name][1](
                                        x, *quarks, phB[mask][frame]
                                    ),
                                ),
                            )
                            secondary_xaxes[index].set_xticks(np.logspace(-20,20,num=41))
                            if index < width or index >= height * width:
                                secondary_xaxes[index].set_xlabel(
                                    f"${general.MASS[mask_name+'-phi']/sym.Abs(general.MASS[mask_name]-general.MASS['phiBare'])}$"
                                )
                    return [l for l, _, _ in plot_elements]

                match quarks:
                    case (1, 1, 1):
                        quark_name = "equal"
                    case (2, -1, -1):
                        quark_name = "charge"
                    case _:
                        quark_name = str(quarks)
                filename = f"{tolerance:.1e}_{quark_name}_{mask_name}"

                for d in [1 - 1e-5, 1 - 1e-1]:
                    for m in [
                        solver.MASS_NEUTRAL_PION,
                        solver.MASS_ETA,
                        solver.MASS_ETA_PRIME,
                    ]:
                        nearness = np.abs(phB[mask] / (d * m) - 1)
                        target = np.argmin(nearness)
                        if nearness[target] > 1e-3:
                            continue
                        update(target)
                        fig_frame.savefig(
                            f"frame_{filename}_{target:0>4n}_{phB[mask][target]:.3e}.pdf"
                        )

                ani = animation.FuncAnimation(
                    fig,
                    update,
                    frames=len(xs[mask]),
                    interval=5000 / len(xs[mask]),
                )
                ani.save(filename=f"animate_{filename}.gif", writer="pillow")
                # ani.save(filename=f"animate_{filename}.mp4", writer="ffmpeg")

                fig.clear()
                fig_frame.clear()
