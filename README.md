`fanpy` is a free and open source Python module for constructing and solving the Schrödinger equation. It is designed to be a research tool for developing new methods that address different aspects of the Schrödinger equation: the Hamiltonian, the wavefunction, the objective that corresponds to the Schrödinger equation, and the algorithm with which the Schrödinger equation is solved. We aim to provide an accessible tool that can facilitate the research between multiple aspects of the electronic structure problem.

Installation
============
To install `fanpy`, use `python setup.py --user` or `pip install --user -e .` from the project home directory.

`fanpy` only uses standard dependencies easily avaialable on most operating systems and that have been stable for the last few years. Non-standard modules are supported optionally, such that the `fanpy` module is still useable without this dependency.

Since compilation is often not trivial for some operating systems, the `fanpy` module will be pure Python (with an optional compilation) to support as many systems as possible.

Features
========
Wavesfunctions
--------------
Following wavefunction ansatze are supported:
- CI single and double excitations (CISD)
- doubly-occupied configuration interaction (DOCI)
- full CI (FCI)
- selected CI wavefunctions with a user-specified set of Slater determinants
- antisymmetrized products of geminals (APG)
- antisymmetrized products of geminals with disjoint orbital sets (APsetG)
- antisymmetrized product of interacting geminals (APIG)
- antisymmetric product of 1-reference-orbital interacting geminals (AP1roG)
- antisymmetric product of rank-two interacting geminals (APr2G)
- determinant ratio wavefunctions
- antisymmetrized products of tetrets (4-electron wavefunctions)
- matrix product states (MPS)
- neural network wavefunctions
- coupled-cluster (CC) with arbitrary excitations (e.g. CCSD, CCSDT)
- CC with seniority-specific excitations)
- pair-CC-doubles (PCCD)
- geminal coupled-cluster wavefunctions (e.g. AP1roGSD, APG1roSD, APsetG1roSD)
- wavefunctions with nonorthogonal orbitals (i.e. VB structures)
- linear combinations of any of the aforementioned wavefunctions

Hamiltonians
------------
Following Hamiltonian are supported:
- Restricted electronic Hamiltonian
- Unrestricted electronic Hamiltonian
- Generalized electronic Hamiltonian
- Seniority zero electronic Hamiltonian
- Restricted Fock operator
- Hamiltonians from [ModelHamiltonian](https://github.com/theochem/ModelHamiltonian) (Parsier-Parr-Pople, Hubbard,
Hückel, Ising, Heisenberg, and Richardson)

Objective
---------
Following objectives are supported:
- energy expectation value
- projected Schrödinger equation
- local energy

Instructions
============
For details on how to use `fanpy` module, please consult the documentation. To get started, see the example script [AP1roG Example](https://github.com/QuantumElephant/fanpy/blob/master/docs/example_ap1rog.rst) and the utility tool for generating scripts [make_script.py](https://github.com/QuantumElephant/fanpy/blob/master/docs/script_make_script.rst)
