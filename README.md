# Pion Rotation Solver

## Requirements

This was tested on Python versions 3.12.3, 3.13.11, and 3.14.2.

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
The basic usage is to first initialize, which gives the argument order for the system of equations:

```python
argument_order, system_of_equations, _ = initialize()
```

You then need to prepare two dictionaries, one representing fixed known quantities, and the other representing unknown quantities and their initial guesses for the solver to start from:

```python
known = {
    rotation.symbolic.MASS["pi"]: 0.1349768,
    rotation.symbolic.MASS["eta"]: 0.547862,
    rotation.symbolic.ANGLE["alpha"]: 0.8,
    rotation.symbolic.ANGLE["beta"]: 0.7,
    rotation.symbolic.ANGLE["gamma"]: 2,
}

unknown = {
    rotation.symbolic.MASS["phi"]: 1,
    rotation.symbolic.MIX["phi"]: 1,
    rotation.symbolic.MIX["pi"]: 0.1349768,
    rotation.symbolic.MIX["eta"]: 0.547862,
    rotation.symbolic.MIX["1x2"]: 1,
    rotation.symbolic.MIX["1x1"]: 1,
}
```

Note that the system of equations requires there to be exactly 6 unknowns, and 5 unknowns.
In addition, the dictionary keys must be the symbols defined by `rotation.symbolic`.
After this, the solver can be called, which will raise a `ValueError` if the solution isn't within tolerance (which defaults to $10^{-8}$):

```python
solved = solve_mixing(system_of_equations, argument_order, unknown, known, tolerance=1e-1)
```

The result is a dictionary of the unknowns you gave it mapped to the solutions the solver found.
The `example.ipynb` shows how to iterate this process multiple times and plot the result.

## Citation

This was used in [arxiv:2602.xxxxx](https://arxiv.org/abs/2602.xxxxx).
