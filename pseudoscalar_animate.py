# %%
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np

from rotation.numerical import solver
from rotation.symbolic import general

# %%
if __name__ == "__main__":
    # storage = pathlib.Path() / "results_pseudo_tol6_hist.pkl"
    # if storage.is_file():
    #     print(f"Loading results from {storage}")
    #     with open(storage, "rb") as f:
    #         results = pickle.load(f)

    # new = {}
    # for xs, saved, err, quarks, tol in results:
    #     if tol not in new:
    #         new[tol] = {}
    #     if quarks not in new[tol]:
    #         new[tol][quarks] = {}
    #     if saved[general.MASS["phiBare"]][0] in new[tol][quarks]:
    #         raise ValueError("Bare Phi mass already present")
    #     solved = {}
    #     for k,v in saved.items():
    #         if r"\tilde{g}" in str(k) or r"E_{" in str(k):
    #             break
    #         solved[k] = v
    #     m_phi_bare = solved[general.MASS["phiBare"]][0]
    #     for k in [general.MASS["phiBare"], general.MASS["pi0"], general.MASS["eta"], general.MASS["etaP"], general.MASS["pi0-phi"], general.MASS["eta-phi"], general.MASS["etaP-phi"]]:
    #         del solved[k]
    #     new[tol][quarks][saved[general.MASS["phiBare"]][0]] = {"x_points": xs}
    #     new[tol][quarks][saved[general.MASS["phiBare"]][0]] |= solved
    #     finite = np.isfinite(new[tol][quarks][saved[general.MASS["phiBare"]][0]][general.ANGLE["alpha"]])
    #     for k,v in new[tol][quarks][saved[general.MASS["phiBare"]][0]].items():
    #         new[tol][quarks][saved[general.MASS["phiBare"]][0]][k] = v[finite]

# %%
    storage_new = pathlib.Path() / "results_pseudo_tol6.pkl"
    # with open(storage_new, "wb") as f:
    #     pickle.dump(new, f)

    with open(storage_new, "rb") as f:
        results = pickle.load(f)

    for tolerance in results:
        for quarks, scan in results[tolerance].items():
            phis = sorted(scan)
            results[tolerance][quarks] = {k: v[np.newaxis, ...] for k,v in scan[phis[0]].items()}
            for p in phis[1:]:
                syn_shape = next(iter(results[tolerance][quarks].values())).shape
                curr_shape = next(iter(scan[p].values())).shape
                if curr_shape[-1] > syn_shape[-1]:
                    for k,v in results[tolerance][quarks].items():
                        results[tolerance][quarks][k] = np.pad(v, ((0,0),(0,curr_shape[-1]-syn_shape[-1])), mode="constant", constant_values=np.nan)
                    syn_shape = *syn_shape[:-1], curr_shape[-1]
                for k,v in scan[p].items():
                    if k in results[tolerance][quarks]:
                        if syn_shape[-1] > curr_shape[-1]:
                            v = np.pad(v, (0,syn_shape[-1]-curr_shape[-1]), mode="constant", constant_values=np.nan)
                        results[tolerance][quarks][k] = np.append(results[tolerance][quarks][k], v[np.newaxis, ...], axis=0)
                    else:
                        raise ValueError("Had an extra key not in others, they should all be the same.")
            results[tolerance][quarks][general.MASS["phiBare"]] = np.array(phis)

# %%
    tab20 = np.array(plt.get_cmap("tab20").colors).reshape(10,2,3)
    for tolerance in results:
        for quarks, scan in results[tolerance].items():
            ph, xs, al, be, ga = scan[general.MASS["phiBare"]], scan["x_points"], scan[general.ANGLE["alpha"]], scan[general.ANGLE["beta"]], scan[general.ANGLE["gamma"]]
            pi = np.abs(ph/solver.MASS_NEUTRAL_PION - 1) < 0.11
            eta = np.abs(ph/solver.MASS_ETA - 1) < 0.11
            etaP = np.abs(ph/solver.MASS_ETA_PRIME - 1) < 0.11
            other = ~pi & ~eta & ~etaP
            mask = ()
            plt.plot(xs[mask].T, al[mask].T, label=ph[mask], color=tab20[0,0])
            plt.plot(xs[mask].T, be[mask].T, label=ph[mask], color=tab20[1,0])
            plt.plot(xs[mask].T, ga[mask].T, label=ph[mask], color=tab20[2,0])
            plt.xscale("log")
            plt.legend()
            # for p,x,a,b,g in zip(ph, xs, al, be, ga):
            #     plt.plot(x, a, label=f"${general.ANGLE['alpha']}$")
            #     plt.plot(x, b, label=f"${general.ANGLE['beta']}$")
            #     plt.plot(x, g, label=f"${general.ANGLE['gamma']}$")
            #     plt.title(f"${general.MASS['phiBare']} = {p}$")
            #     plt.legend()
            #     plt.show()
            break
        break
