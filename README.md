# Rotation Solver

## Requirements

This was tested on Python versions 3.10 through 3.14.

To run the rotation solver you'll need and make plots

- Matplotlib
- Numpy
- Scipy
- Sympy

Optionally to run the notebook also install

- Jupyter

Both lists with the tested package versions are found in the `requirements.txt` and `requirements-optional.txt` files, which can be used with

```shell
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

to install the packages either in your global Python environment or a virtual Python environment.

## Usage

The notebooks contain examples of how to use the rotation solver.
The basic usage is to first initialize the symbolic equation as a system of equations and the argument order when calling an equation:

```python
# Pseudo scalar mediator case
import rotation.symbolic.general
import rotation.symbolic.pseudo
import rotation.numerical.solver

rotation_matrix = rotation.symbolic.general.create_4d_rotation()[0]
pseudo_mixing_equation = rotation.symbolic.pseudo.create_mix_equation(rotation_matrix)[0]
argument_order, system_of_equations = rotation.numerical.solver.initialize_system(pseudo_mixing_equation.lhs - pseudo_mixing_equation.rhs)
```

```python
# Axial vector mediator case
import rotation.symbolic.general
import rotation.symbolic.axial
import rotation.numerical.solver

rotation_matrix = rotation.symbolic.general.create_3d_rotation()[0]
kinetic_rotation_matrix = rotation.symbolic.axial.create_kinetic()
axial_mixing_equation = rotation.symbolic.axial.create_axial_mix_equation(rotation_matrix, kinetic_rotation_matrix)[0]
argument_order, system_of_equations = rotation.numerical.solver.initialize_system(axial_mixing_equation.lhs - axial_mixing_equation.rhs)
```

You then need to prepare two dictionaries, one representing fixed known quantities, and the other representing unknown quantities and their initial guesses for the solver to start from:

```python
# Pseudo scalar mediator case
import rotation.numerical.pseudo
bare_masses = rotation.numerical.pseudo.pseudo_calculate_bare_masses(
    coupling_up=2e-2,
    coupling_down=-1e-2,
    coupling_strange=-1e-2,
)
known = {
    rotation.symbolic.general.MASS["pi0"]: 0.1349768,
    rotation.symbolic.general.MASS["eta"]: 0.547862,
    rotation.symbolic.general.MASS["phiBare"]: 0.1,
    rotation.symbolic.general.MASS["pi0-phi"]: bare_masses[rotation.symbolic.general.MASS["pi0-phi"]],
    rotation.symbolic.general.MASS["eta-phi"]: bare_masses[rotation.symbolic.general.MASS["eta-phi"]],
    rotation.symbolic.general.MASS["etaP"]: 0.95778,
    rotation.symbolic.general.MASS["etaP-phi"]: bare_masses[rotation.symbolic.general.MASS["etaP-phi"]],
}
unknown = {
    rotation.symbolic.general.MASS["phi"]: 0.1,
    rotation.symbolic.general.MASS["pi0Bare"]: 0.1349768,
    rotation.symbolic.general.MASS["etaBare"]: 0.547862,
    rotation.symbolic.general.ANGLE["alpha"]: 0.0,
    rotation.symbolic.general.ANGLE["beta"]: 0.0,
    rotation.symbolic.general.ANGLE["gamma"]: 0.0,
    rotation.symbolic.general.MASS["etaPBare"]: 0.95778,
    rotation.symbolic.general.ANGLE["delta"]: 0.0,
    rotation.symbolic.general.ANGLE["zeta"]: 0.0,
    rotation.symbolic.general.ANGLE["kappa"]: 0.0,
}
```

Extra note: for the axial vector mediator, the kinetic mixing angles can ether be provided to the solver, or calculated from known particle masses as input.

```python
# Axial vector mediator case
import rotation.numerical.axial
bare_masses = rotation.numerical.axial.axial_calculate_bare_masses(
    coupling_up=2e-1,
    coupling_down=-1e-1,
    coupling_strange=-1e-1,
)
kin_u, kin_v, kin_w = rotation.numerical.axial.calculate_uvw(
    m_axial=0.1,
    m_pi0_axial=bare_masses[rotation.symbolic.general.MASS["pi0-AP"]],
    m_eta_axial=bare_masses[rotation.symbolic.general.MASS["eta-AP"]],
    m_etap_axial=bare_masses[rotation.symbolic.general.MASS["etaP-AP"]],
)
known = {
    rotation.symbolic.general.MASS["pi0"]: 0.1349768,
    rotation.symbolic.general.MASS["eta"]: 0.547862,
    rotation.symbolic.general.MASS["etaP"]: 0.95778,
    rotation.symbolic.general.ANGLE["u"]: kin_u,
    rotation.symbolic.general.ANGLE["v"]: kin_v,
    rotation.symbolic.general.ANGLE["w"]: kin_w,
}
unknown = {
    rotation.symbolic.general.MASS["pi0Bare"]: 0.1349768,
    rotation.symbolic.general.MASS["etaBare"]: 0.547862,
    rotation.symbolic.general.MASS["etaPBare"]: 0.95778,
    rotation.symbolic.general.ANGLE["alpha"]: 0.0,
    rotation.symbolic.general.ANGLE["beta"]: 0.0,
    rotation.symbolic.general.ANGLE["gamma"]: 0.0,
}
```

These dictionary keys must be the symbols defined by `rotation.symbolic.general`.
After this, the solving function can be created and the solver can be called, which will raise a `ValueError` if the solution isn't within tolerance (which defaults to $10^{-8}$):

```python
# Create function
func = rotation.numerical.solver.create_func(
    system_of_equations,
    argument_order,
    known,
)
# Solve system
solved = rotation.numerical.solver.solve_mixing(
    func,
    argument_order,
    unknown,
)
```

The result is a dictionary of the unknowns you gave it mapped to the solutions the solver found.
The `example.ipynb` shows how to iterate this process multiple times and plot the result.

## Scanning Examples

The following show animated examples after sweeping the mediator mass in a scan, performed by the `*_scan.py` scripts.
The plotting is then performed by the `*_animate.py` scripts.
The data from `*_scan.py` was used in [arxiv:2611.xxxxx](https://arxiv.org/abs/2611.xxxxx), and is available at [![DOI](https://zenodo.org/badge/23220809.svg)](https://zenodo.org/badge/latestdoi/23220809).

### Charge proportional quark couplings:

![](figures/animate_pseudo_1.0e-08_charge_other.gif)
![](figures/animate_pseudo_1.0e-08_charge_pi0.gif)
![](figures/animate_axial_1.0e-08_charge_other.gif)
![](figures/animate_axial_1.0e-08_charge_pi0.gif)

### Equal proportional quark couplings:

![](figures/animate_pseudo_1.0e-08_equal_other.gif)
![](figures/animate_pseudo_1.0e-08_equal_eta.gif)
![](figures/animate_axial_1.0e-08_equal_other.gif)
![](figures/animate_axial_1.0e-08_equal_eta.gif)

## Citation

[![DOI](https://zenodo.org/badge/.svg)](https://zenodo.org/badge/latestdoi/)
This was used in [arxiv:2611.xxxxx](https://arxiv.org/abs/2611.xxxxx).
The equations within ...
The scanned points can be found in the Zenodo entry: [![DOI](https://zenodo.org/badge/23220809.svg)](https://zenodo.org/badge/latestdoi/23220809)
