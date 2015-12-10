"""
Interface to HORTON's AP1roG module.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from horton import context, IOData
from horton.cext import compute_nucnuc
from horton.correlatedwfn import RAp1rog
from horton.gbasis import get_gobasis
from horton.matrix import DenseLinalgFactory
from horton.meanfield import AufbauOccModel
from horton.meanfield import guess_core_hamiltonian, PlainSCFSolver, REffHam
from horton.meanfield.observable import RDirectTerm, RExchangeTerm, RTwoIndexTerm
from horton.orbital_utils import transform_integrals


def ap1rog_from_horton(fn=None, basis=None, npairs=None, guess="apig"):
    """
    Compute information about a molecule's AP1roG wavefunction with HORTON.

    Parameters
    ----------
    fn : str
        The file containing the molecular geometry.
    basis: str
        The basis set to use for the orbitals.
    npairs : int
        The number of occupied orbitals.
    guess : str, optional
        The type of geminal coefficient matrix guess to return.  One of "apig" or
        "ap1rog".  Defaults to "apig".

    Returns
    -------
    result : dict
        Contains "mol", an IOData instance; "basis", a GOBasis instance; and "ham", a
        tuple containing the terms of the Hamiltonian matrix.

    """

    # Load the molecule and basis set from file
    mol = IOData.from_file(context.get_fn(fn))
    obasis = get_gobasis(mol.coordinates, mol.numbers, basis)

    # Fill in the orbital expansion and overlap
    occ_model = AufbauOccModel(npairs)
    lf = DenseLinalgFactory(obasis.nbasis)
    orb = lf.create_expansion(obasis.nbasis)
    olp = obasis.compute_overlap(lf)

    # Construct Hamiltonian
    kin = obasis.compute_kinetic(lf)
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
    two = obasis.compute_electron_repulsion(lf)
    external = {"nn": compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        RTwoIndexTerm(kin, "kin"),
        RDirectTerm(two, "hartree"),
        RExchangeTerm(two, "x_hf"),
        RTwoIndexTerm(na, "ne"),
    ]
    ham = REffHam(terms, external)
    guess_core_hamiltonian(olp, kin, na, orb)

    # Do Hartree-Fock SCF
    PlainSCFSolver(1.0e-6)(ham, lf, olp, occ_model, orb)

    # Get initial guess at energy, coefficients from AP1roG
    one = kin
    one.iadd(na)
    ap1rog = RAp1rog(lf, occ_model)
    energy, cblock = ap1rog(one, two, external["nn"], orb, olp, False)

    # Transform the one- and two- electron integrals into the MO basis
    one_mo, two_mo = transform_integrals(one, two, "tensordot", orb)

    # RAp1rog only returns the "A" block from the [I|A]-shaped coefficient matrix
    guess = guess.lower()
    if guess == "apig":
        coeffs = np.eye(npairs, obasis.nbasis)
        coeffs[:, npairs:] += cblock._array
    elif guess == "ap1rog":
        coeffs = cblock._array
    else:
        raise NotImplementedError

    return {
        "mol": mol,
        "basis": obasis,
        "ham": (one_mo[0]._array, two_mo[0]._array, external["nn"]),
        "energy": energy,
        "coeffs": coeffs,
    }

# vim: set textwidth=90 :
