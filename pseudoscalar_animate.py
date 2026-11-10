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
rcParams.update(
    {
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,  # cmr10 has no Unicode minus
        "font.size": 17 * FONT_SCALE,
        "axes.labelsize": 24 * FONT_SCALE,
        "axes.titlesize": 24 * FONT_SCALE,
        "xtick.labelsize": 19 * FONT_SCALE,
        "ytick.labelsize": 19 * FONT_SCALE,
        "legend.fontsize": 19 * FONT_SCALE,
    }
)


def small_alpha(m_phi, m_pi0_phi):
    return 0.5 * m_pi0_phi**2 / (solver.MASS_NEUTRAL_PION**2 - m_phi**2)


def small_beta(m_phi, m_eta_phi):
    return 0.5 * m_eta_phi**2 / (solver.MASS_ETA**2 - m_phi**2)


def small_gamma(m_phi, m_etaP_phi):
    return 0.5 * m_etaP_phi**2 / (solver.MASS_ETA_PRIME**2 - m_phi**2)


def pseudo_mass_ratio():
    sx, sr = sym.symbols("x r", nonnegative=True)
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
    for label in ("pi0", "eta", "etaP"):
        ratio = bares[general.MASS[f"{label}-phi"]] ** 2 / sym.Abs(
            general.MASS[label] ** 2 - general.MASS["phiBare"] ** 2
        )
        ratio = ratio.subs(parameter)
        strengths = [_x[sx] for _x in sym.solve(sr - ratio, [sx], dict=True)]
        result[label] = (
            sym.lambdify([sx] + call, ratio),
            [([sr] + call, st) for st in strengths],
        )
    return result


def _generic_plot(
    ax,
    x_strength: np.NDArray,
    draw_order: list[tuple[np.NDArray, dict]],
    y_min_limit: float = None,
    y_max_limit: float = None,
):
    limits = [np.nanmin(draw_order[0][0]), np.nanmax(draw_order[0][0])]
    for ys, _ in draw_order[1:]:
        limits[0] = min(limits[0], np.nanmin(ys))
        limits[1] = max(limits[1], np.nanmax(ys))
    if y_min_limit is not None:
        limits[0] = max(limits[0], y_min_limit)
    if y_max_limit is not None:
        limits[1] = min(limits[1], y_max_limit)
    ax.set_ylim(*limits)
    ax.set_xscale("log")
    elements = []
    for ys, style in draw_order:
        if np.all(np.abs(ys[~np.isnan(ys)]) < 1e-14):
            continue  # if it's exactly zero, we don't need to show it
        elements.append((ax.plot(x_strength[0], ys[0], **style)[0], x_strength, ys))
    return elements


def draw_angles(
    ax,
    x_strength: np.NDArray,
    draw_order: list[tuple[np.NDArray, dict]],
):
    elements = _generic_plot(
        ax,
        x_strength,
        draw_order,
        y_min_limit=-1 / 2 * np.pi,
        y_max_limit=1 / 2 * np.pi,
    )
    ax.set_ylabel(ylabel="Angle [Rad]")
    sec_y = ax.secondary_yaxis(
        "right", functions=(lambda x: x * 180 / np.pi, lambda x: x * np.pi / 180)
    )
    sec_y.set_ylabel("Angle [Deg]")
    ax.legend(loc="lower left")
    return elements


def draw_masses(
    ax,
    x_strength: np.NDArray,
    draw_order: list[tuple[np.NDArray, dict]],
    m_mediator: tuple[np.NDArray, dict] = None,
    m_pi0: tuple[np.NDArray, dict] = None,
    m_eta: tuple[np.NDArray, dict] = None,
    m_etaP: tuple[np.NDArray, dict] = None,
):
    elements = []
    if m_mediator is not None:
        elements.append(
            (
                ax.axhline(m_mediator[0][0], **m_mediator[1]),
                None,
                m_mediator[0][..., np.newaxis],
            )
        )
    elements += _generic_plot(
        ax, x_strength, draw_order, y_min_limit=1e-3, y_max_limit=1e2
    )
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
        ylabel="Mass [GeV]",
    )
    ax.legend(loc="lower left")
    return elements


def draw_mesons(
    ax,
    x_strength: np.NDArray,
    draw_order: list[tuple[np.NDArray, dict]],
    bare_coupling: tuple[float, dict],
):
    if bare_coupling is not None:
        ax.axhline(bare_coupling[0], **bare_coupling[1])
    elements = _generic_plot(
        ax,
        x_strength,
        draw_order,
        y_min_limit=-bare_coupling[0],
        y_max_limit=bare_coupling[0],
    )
    ax.set_ylabel("Coupling")
    ax.legend(loc="lower left")
    return elements


def draw_ratios(
    ax,
    x_strength: np.NDArray,
    draw_order: list[tuple[np.NDArray, dict]],
):
    elements = _generic_plot(
        ax, x_strength, draw_order, y_min_limit=-10, y_max_limit=10
    )
    ax.set_ylabel("Approx / Exact")
    ax.legend(loc="lower left")
    return elements


def draw_fractions(
    ax,
    x_strength: np.NDArray,
    draw_order: list[tuple[np.NDArray, dict]],
):
    elements = _generic_plot(
        ax, x_strength, draw_order, y_min_limit=1e-16, y_max_limit=1
    )
    ax.set(
        yscale="log",
        ylabel="Abs(Approx / Exact - 1)",
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
    for tolerance, _ in stored.items():
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
    for tolerance, _ in stored.items():
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
    height, width, figsize = 2, 2, np.array((7.8, 5.8))
    secondaries = pseudo_mass_ratio()
    for tolerance, _ in stored.items():
        for quarks in stored[tolerance]:
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
                    figsize=figsize * 1.1 * np.array((width, height)),
                )
                fig_frame, ax_frame = plt.subplots(1, 1, figsize=figsize)
                limits = np.empty((*axs.shape, 2))

                if quarks == (2, -1, -1) and mask_name == "pi0":
                    meson_couplings = ("pi0-chi", "eta-chi", "phi-gamma")
                else:
                    meson_couplings = ("pi0-chi", "eta-chi", "etaP-chi", "phi-gamma")

                plot_elements = []
                plot_elements += draw_angles(
                    axs[0, 0],
                    xs[mask],
                    [
                        (
                            stored[tolerance][quarks][k][mask],
                            {
                                "label": f"${k}$",
                                "color": angle_style[k][0],
                                "linestyle": "-.",
                            },
                        )
                        for k in general.ANGLE.values()
                        if k in stored[tolerance][quarks]
                    ]
                    + [
                        (
                            approx[tolerance][quarks][k][mask],
                            {
                                "label": f"${k}_\\text{{approx}}$",
                                "color": angle_style[k][1],
                                "linewidth": 1,
                            },
                        )
                        for k in general.ANGLE.values()
                        if k in approx[tolerance][quarks]
                    ],
                )

                plot_elements += draw_masses(
                    axs[0, 1],
                    xs[mask],
                    [
                        (
                            stored[tolerance][quarks][general.MASS[k]][mask],
                            {
                                "label": f"${general.MASS[k]}$",
                                "color": mass_style[general.MASS[k]][0],
                                "linestyle": "-.",
                            },
                        )
                        for k in ["phi", "pi0Bare", "etaBare", "etaPBare"]
                    ],
                    m_mediator=(phB[mask], {"color": tab20[7, 1], "linewidth": 1}),
                    **{
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
                    [
                        (
                            stored[tolerance][quarks][general.COUPLING[k]][mask],
                            {
                                "label": f"${general.COUPLING[k]}$",
                                "color": meson_style[general.COUPLING[k]][0],
                                "linestyle": "-.",
                            },
                        )
                        for k in [
                            "phi-chi",
                            *meson_couplings,
                            "phi-gamma",
                        ]
                    ]
                    + [
                        (
                            approx[tolerance][quarks][general.COUPLING[k]][mask],
                            {
                                "label": "$"
                                + str(general.COUPLING[k])[:-1]
                                + r", approx}$",
                                "color": meson_style[general.COUPLING[k]][1],
                            },
                        )
                        for k in [*meson_couplings, "phi-gamma"]
                    ],
                    bare_coupling=(
                        g_chi,
                        {
                            "label": f"${general.COUPLING['chi']} = {g_chi}$",
                            "color": tab20[7, 1],
                            "linewidth": 1,
                        },
                    ),
                )

                plot_elements += draw_ratios(
                    axs[1, 1],
                    xs[mask],
                    [
                        (
                            ratio[tolerance][quarks][general.ANGLE[k]][mask],
                            {
                                "label": f"${general.ANGLE[k]}$",
                                "color": angle_style[general.ANGLE[k]][0],
                            },
                        )
                        for k in ["alpha", "beta", "gamma"]
                    ]
                    + [
                        (
                            ratio[tolerance][quarks][general.MASS[k]][mask],
                            {
                                "label": f"${general.MASS[k]}$",
                                "color": mass_style[general.MASS[k]][0],
                                "linestyle": "-.",
                            },
                        )
                        for k in ["pi0Bare", "etaBare", "etaPBare"]
                    ],
                )

                for ax in [ax_frame]:
                    plot_elements += draw_fractions(
                        ax,
                        xs[mask],
                        [
                            (
                                fraction[tolerance][quarks][general.COUPLING[k]][mask],
                                {
                                    "label": f"${general.COUPLING[k]}$",
                                    "color": meson_style[general.COUPLING[k]][0],
                                },
                            )
                            for k in meson_couplings
                        ]
                        + [
                            (
                                fraction[tolerance][quarks][general.MASS["phi"]][mask],
                                {
                                    "label": f"${general.MASS['phi']}$",
                                    "color": mass_style[general.MASS["phi"]][0],
                                },
                            )
                        ],
                    )

                for ax in list(axs[-1, :]) + [ax_frame]:
                    ax.set_xlabel("Q")
                match mask_name:
                    case "other":
                        xlims = (3e-5, 3e2)
                    case "pi0":
                        xlims = (2e-6, 2e-1)
                    case "eta":
                        xlims = (3e-6, 3e-1)
                    case "etaP":
                        xlims = (5e-4, 5e-1)
                    case _:
                        xlims = (np.nanmin(xs), np.nanmax(xs))
                secondary_xaxes = []
                if mask_name in secondaries:
                    to_secondary = secondaries[mask_name][0]
                    from_secondary = None
                    for call, sol in secondaries[mask_name][1]:
                        sol = sol.subs(
                            {general.COUPLING[k]: q for k, q in zip("uds", quarks)}
                        )
                        if sol.is_nonnegative:
                            from_secondary = sym.lambdify(call, sol)
                            break
                for index, ax in enumerate(list(axs.flat) + [ax_frame]):
                    ax.set_xlim(*xlims)
                    ax.grid()
                    ax.set_xscale("log")
                    if mask_name in secondaries and from_secondary is not None:
                        secondary_xaxes.append(
                            ax.secondary_xaxis(
                                "top",
                                functions=(
                                    lambda x: to_secondary(x, *quarks, phB[mask][0]),
                                    lambda x: from_secondary(x, *quarks, phB[mask][0]),
                                ),
                            )
                        )
                        secondary_xaxes[-1].set_xticks(np.logspace(-20, 20, num=41))
                        if index < width or index >= height * width:
                            secondary_xaxes[-1].set_xlabel(
                                f"${general.MASS[mask_name+'-phi']}^{{2}} / Abs({general.MASS[mask_name]}^{{2}} - {general.MASS['phiBare']}^{{2}})$",
                                labelpad=16,
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
                    # Place vertical line where the first angle crosses 0.3
                    all_angles = np.concatenate(
                        [
                            stored[tolerance][quarks][k][mask][..., np.newaxis]
                            for k in general.ANGLE.values()
                            if k in stored[tolerance][quarks]
                        ],
                        axis=-1,
                    )
                    greater = np.abs(all_angles) > 0.3
                    lesser = np.abs(all_angles) < 0.3
                    rising = greater[..., :-1, :] & lesser[..., 1:, :]
                    falling = lesser[..., :-1, :] & greater[..., 1:, :]
                    before, after = np.zeros_like(
                        all_angles, dtype=bool
                    ), np.zeros_like(all_angles, dtype=bool)
                    before[..., :-1, :] = rising | falling
                    after[..., 1:, :] = rising | falling
                    dx = np.nan * np.empty_like(all_angles, dtype=float)
                    dy = np.nan * np.empty_like(all_angles, dtype=float)
                    x_lerp = np.nan * np.empty_like(all_angles, dtype=float)
                    all_xs = np.repeat(
                        xs[mask][:, :, np.newaxis], all_angles.shape[2], axis=2
                    )
                    dx[before] = all_xs[after] - all_xs[before]
                    dy[before] = np.abs(all_angles[after]) - np.abs(all_angles[before])
                    x_lerp[before] = (
                        dx[before] / dy[before] * (0.3 - np.abs(all_angles[before]))
                        + all_xs[before]
                    )
                    is_thresh = np.abs(all_angles) == 0.3
                    x_lerp[is_thresh] = all_xs[is_thresh]
                    vert_x = np.nanmin(
                        x_lerp, axis=(1, 2)
                    )  # We're saving the 0.3 of the minimum x value between all 0.3 crossings and all angles
                    plot_elements.append(
                        (
                            ax.axvline(vert_x[0], color="black", linestyle=":"),
                            vert_x[..., np.newaxis],
                            None,
                        )
                    )
                    # Hatch out the region where solver fails
                    maxes = np.nanmax(xs[mask], axis=1)
                    lims = ax.get_xlim()[1] * np.ones_like(maxes)
                    bad_xs = np.concatenate(
                        (maxes[:, np.newaxis], lims[:, np.newaxis]), axis=1
                    )
                    bad_ys = ax.get_ylim()
                    plot_elements.append(
                        (
                            ax.fill_between(
                                bad_xs[0], *bad_ys, facecolor="red", alpha=1 / 3
                            ),
                            bad_xs,
                            bad_ys,
                        )
                    )

                fig.suptitle(f"${general.MASS['phiBare']} = (10^{{00}})$")
                fig.tight_layout()

                fig_frame.suptitle(f"${general.MASS['phiBare']} = (10^{{00}})$")
                fig_frame.tight_layout()

                def update(frame):
                    for line, x, y in plot_elements:
                        try:
                            if x is not None:
                                line.set_xdata(x[frame])
                            if y is not None:
                                line.set_ydata(y[frame])
                        except AttributeError:
                            line.set_data(x[frame], *y)

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
                    if (
                        mask_name in secondaries
                        and from_secondary is not None
                        and phB[mask][frame]
                        not in [
                            solver.MASS_NEUTRAL_PION,
                            solver.MASS_ETA,
                            solver.MASS_ETA_PRIME,
                        ]
                    ):
                        for index, ax in enumerate(list(axs.flat) + [ax_frame]):
                            secondary_xaxes[index].remove()
                            secondary_xaxes[index] = ax.secondary_xaxis(
                                "top",
                                functions=(
                                    lambda x: to_secondary(
                                        x, *quarks, phB[mask][frame]
                                    ),
                                    lambda x: from_secondary(
                                        x, *quarks, phB[mask][frame]
                                    ),
                                ),
                            )
                            secondary_xaxes[index].set_xticks(
                                np.logspace(-20, 20, num=41)
                            )
                            if index < width or index >= height * width:
                                secondary_xaxes[index].set_xlabel(
                                    f"${general.MASS[mask_name+'-phi']}^{{2}} / Abs({general.MASS[mask_name]}^{{2}} - {general.MASS['phiBare']}^{{2}})$",
                                    labelpad=16,
                                )
                    return [l for l, _, _ in plot_elements]

                match quarks:
                    case (1, 1, 1):
                        quark_name = "equal"
                    case (2, -1, -1):
                        quark_name = "charge"
                    case _:
                        quark_name = str(quarks)
                filename = f"pseudo_{tolerance:.1e}_{quark_name}_{mask_name}"

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
