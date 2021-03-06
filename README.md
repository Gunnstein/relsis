# Relsis

A package to perform reliability and sensitivity analysis in python (relsis). The package provides an interface to define random variables and includes solvers to evaluate limit state functions. The solvers include FORM and Monte Carlo simulations with and without enhanced sampling schemes. The Monte Carlo implementation allows for defining correlated variables.

## Installation

Download to your computer, create a wheel and install the wheel, e.g:

```sh
git clone https://github.com/gunnstein/relsis.git
cd relsis
pip install .
```

## Usage

The package includes modules for:

- Crude and stratified (LHS, Sobol seq.) sampling of probability distributions.
- Analyzing functions (Monte Carlo simulations and Most probable point (FORM))
- Performing sensitivity analysis (Alpha factors, Method of Morris, Sobol variance decomposition)

The example (example.py) is a good place to start to get into the use of the package.

## Acknowledgements

The code of `sampling/_sobol_seq` module is mainly authored by Corrado Chisari and John Burkardt and distributed under the MIT license. Their contribution is gratefully acknowledged. Please check out the original implementation at https://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html 

## Support

Please [open an issue](https://github.com/Gunnstein/relsis/issues/new) for support.

## Contributing

Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/Gunnstein/relsis/compare/).
