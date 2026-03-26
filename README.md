# Rotation Solver

## Requirements

This was tested on Python versions 3.10 through 3.14.

To run the rotation solver you'll need

- Numpy
- Scipy
- Sympy

Optionally to run the notebook also install

- Jupyter
- Matplotlib

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

rotation_matrix = rotation.symbolic.general.create_3d_rotation()[0]
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
piphi, etaphi = rotation.numerical.pseudo.pseudo_calculate_bare_masses(
    coupling_up=0.7,
    coupling_down=0.1,
    coupling_strange=-0.2,
)
known = {
    rotation.symbolic.general.MASS["pi0"]: 0.1349768,
    rotation.symbolic.general.MASS["eta"]: 0.547862,
    rotation.symbolic.general.MASS["phiBare"]: 2.5,
    rotation.symbolic.general.MASS["pi0-phi"]: piphi,
    rotation.symbolic.general.MASS["eta-phi"]: etaphi,
}
unknown = {
    rotation.symbolic.general.MASS["phi"]: 2.5,
    rotation.symbolic.general.MASS["pi0Bare"]: 0.1349768,
    rotation.symbolic.general.MASS["etaBare"]: 0.547862,
    rotation.symbolic.general.ANGLE["alpha"]: 0.0,
    rotation.symbolic.general.ANGLE["beta"]: 0.0,
    rotation.symbolic.general.ANGLE["gamma"]: 0.0,
}
```

Extra note: for the axial vector mediator, the kinetic mixing angles can ether be provided to the solver, or calculated from known particle masses as input.

```python
# Axial vector mediator case
import rotation.numerical.axial
piA, etaA, etaPA = rotation.numerical.axial.axial_calculate_bare_masses(
    coupling_up=0.7,
    coupling_down=0.1,
    coupling_strange=-0.2,
)
kin_u, kin_v, kin_w = rotation.numerical.axial.calculate_uvw(
    m_axial=1.0,
    m_pi0_axial=piA,
    m_eta_axial=etaA,
    m_etap_axial=etaPA,
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

## Citation

This was used in [arxiv:2603.xxxxx](https://arxiv.org/abs/2602.xxxxx).
