""" Parent class of CI wavefunctions

This module describes wavefunction that are expressed as linear combination of Slater determinants.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.optimize import least_squares
from .wavefunction import Wavefunction
from ..backend import slater
from ..hamiltonian.ci_matrix import ci_matrix
from ..backend.sd_list import sd_list
from ..hamiltonian.density import density_matrix
# FIXME: inherit docstring

__all__ = []


class CIWavefunction(Wavefunction):
    """ Wavefunction expressed as a linear combination of Slater determinants

    Contains the necessary information to variationally solve the CI wavefunction

    Attributes
    ----------
    nelec : int
        Number of electrons
    one_int : 1- or 2-tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        1-tuple for spatial (restricted) and generalized orbitals
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
    two_int : 1- or 3-tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        In physicist's notation
        1-tuple for spatial (restricted) and generalized orbitals
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components)
    dtype : {np.float64, np.complex128}
        Numpy data type
    nuc_nuc : float
        Nuclear-nuclear repulsion energy
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    dict_exc_index : dict from int to int
        Dictionary from the excitation order to the column index of the coefficient matrix
    spin : float
        Total spin of the wavefunction
        Default is no spin (all spins possible)
        0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc
        Positive spin means that there are more alpha orbitals than beta orbitals
        Negative spin means that there are more beta orbitals than alpha orbitals
    civec : tuple of int
        List of Slater determinants used to construct the CI wavefunction

    Properties
    ----------
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals

    Method
    ------
    __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_nuc_nuc(self, nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    assign_excs(self, excs=None)
        Assigns excitations to include in the calculation
    assign_spin(self, spin=None)
        Assigns the spin of the wavefunction
    assign_civec(self, civec=None)
        Assigns the tuple of Slater determinants used in the CI wavefunction
    get_energy(self, include_nuc=True, exc_lvl=0)
        Gets the energy of the CI wavefunction
    compute_density_matrix(self, exc_lvl=0, is_chemist_notation=False, val_threshold=0)
        Constructs the one and two electron density matrices for the given excitation level
    compute_ci_matrix(self)
        Returns CI Hamiltonian matrix in the Slater determinant basis
    generate_civec
        Generates a list of Slater determinants
    """
    def __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None,
                 excs=None, civec=None, spin=None, seniority=None):
        """ Initializes a wavefunction

        Parameters
        ----------
        nelec : int
            Number of electrons

        one_int : np.ndarray(K,K), 1- or 2-tuple np.ndarray(K,K)
            One electron integrals
            For spatial and generalized orbitals, np.ndarray or 1-tuple of np.ndarray
            For unretricted spin orbitals, 2-tuple of np.ndarray

        two_int : np.ndarray(K,K,K,K), 1- or 3-tuple np.ndarray(K,K,K,K)
            For spatial and generalized orbitals, np.ndarray or 1-tuple of np.ndarray
            For unrestricted orbitals, 3-tuple of np.ndarray

        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type
            Default is `np.float64`

        nuc_nuc : {float, None}
            Nuclear nuclear repulsion value
            Default is `0.0`

        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbital used in obtaining the one-electron and two-electron integrals
            Default is `'restricted'`

        excs : list/tuple of int
            Tuple of excitation orders that are relevant to the wavefunction

        civec : iterable of int
            List of Slater determinants used to construct the CI wavefunction

        spin : float
            Total spin of the wavefunction
            Default is no spin (all spins possible)
            0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc
            Positive spin means that there are more alpha orbitals than beta orbitals
            Negative spin means that there are more beta orbitals than alpha orbitals

        seniority : int
            Seniority of the wavefunction
            Default is no seniority (all seniority possible)
        """
        super(CIWavefunction, self).__init__(nelec, one_int, two_int, dtype=dtype, nuc_nuc=nuc_nuc,
                                             orbtype=orbtype)
        self.assign_spin(spin=spin)
        self.assign_seniority(seniority=seniority)
        self.assign_civec(civec=civec)
        self.assign_excs(excs=excs)
        self.sd_coeffs = np.zeros((len(self.civec), len(self.dict_exc_index)))
        self.energies = np.zeros(len(self.dict_exc_index))

    ######################
    # Assignment methods #
    ######################
    def assign_spin(self, spin=None):
        """ Sets the spin of the wavefunction

        Parameters
        ----------
        spin : float
            Total spin of the wavefunction
            Default is no spin (all spins possible)
            0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc
            Positive spin means that there are more alpha orbitals than beta orbitals
            Negative spin means that there are more beta orbitals than alpha orbitals

        Raises
        ------
        TypeError
            If the spin is not an integer, float, or None
        """
        if not isinstance(spin, (int, float, type(None))):
            raise TypeError('Invalid spin of the wavefunction')
        self.spin = spin


    def assign_seniority(self, seniority=None):
        """ Sets the seniority of the wavefunction

        Parameters
        ----------
        seniority : int
            Seniority of the wavefunction
            Default is no seniority (all seniority possible)

        Raises
        ------
        TypeError
            If the seniority is not a float or None
        """
        if not isinstance(seniority, (int, type(None))):
            raise TypeError('Invalid seniority of the wavefunction')
        self.seniority = seniority


    def assign_civec(self, civec=None):
        """ Sets the Slater determinants used in the wavefunction

        Parameters
        ----------
        civec : iterable of int
            List of Slater determinants (in the form of integers that describe the occupation as a
            bitstring)

        Raises
        ------
        TypeError
            If civec is not iterable
            If a Slater determinant cannot be turned into the internal form
        ValueError
            If Slater determinant does not have the right number of electrons
            If there are no Slater determinants that has the given spin
        """
        if civec is None:
            civec = self.generate_civec()

        if not hasattr(civec, '__iter__'):
            raise TypeError("Slater determinants must be given as an iterable")

        filtered_sds = []
        for slater_d in civec:
            slater_d = slater.internal_sd(slater_d)
            if slater.total_occ(slater_d) != self.nelec:
                raise ValueError('Slater determinant, {0}, does not have the right number of'
                                 ' electrons'.format(bin(slater_d)))
            if self.spin is not None and slater.get_spin(slater_d, self.nspatial) != self.spin:
                continue
            if (self.seniority is not None and
                    slater.get_seniority(slater_d, self.nspatial) != self.seniority):
                continue
            filtered_sds.append(slater_d)

        # check if empty
        if len(filtered_sds) == 0:
            terms = {'spin':self.spin, 'seniority':self.seniority}
            end_phrase = ', and '.join('{0}, {1}'.format(i, j) for i, j in terms.iteritems()
                                       if j is not None)
            raise ValueError('Could not find any Slater determinant that has {0}'
                             ''.format(end_phrase))
        self.civec = tuple(filtered_sds)


    def assign_excs(self, excs=None):
        """ Sets orders of excitations to include during calculation

        Parameters
        ----------
        excs : list/tuple of ints
            Orders of excitations to calculate
            By default, only the ground state (0th excitation) is calculated

        Raises
        ------
        TypeError
            If excs is not given as a list/tuple of integers
        ValueError
            If any excitation order is less than 0 or greater than the number of Slater determinants
        """
        if excs is None:
            excs = [0]
        if not isinstance(excs, (list, tuple)) or any(not isinstance(i, int) for i in excs):
            raise TypeError('Orders of excitations must be given as a list or tuple of integers')
        if not all(0 <= exc < len(self.civec) for exc in excs):
            raise ValueError('All excitation orders must be greater than or equal to 0 and less'
                             ' the number of Slater determinants, {0}'.format(len(self.civec)))
        self.dict_exc_index = {exc:i for i, exc in enumerate(excs)}


    ##########
    # Getter #
    ##########
    def get_energy(self, include_nuc=True, exc_lvl=0):
        """ Returns the energy of the system

        Parameters
        ----------
        include_nuc : bool
            Flag to include nuclear nuclear repulsion
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            `n`is the `n`th order excitation

        Returns
        -------
        energy : float
            Total energy if include_nuc is True
            Electronic energy if include_nuc is False

        Raises
        ------
        ValueError
            If the excitation level was not included in the initialization (or in the assignment of
            self.dict_exc_index)
        """
        if exc_lvl not in self.dict_exc_index:
            raise ValueError('Unsupported excitation level, {0}'.format(exc_lvl))
        nuc_nuc = self.nuc_nuc if include_nuc else 0.0
        return self.energies[self.dict_exc_index[exc_lvl]] + nuc_nuc


    ##################
    # Density Matrix #
    ##################
    def compute_density_matrix(self, exc_lvl=0, is_chemist_notation=False, val_threshold=0):
        """ Returns the first and second order density matrices

        Second order density matrix uses the Physicist's notation:
        ..math::
            \Gamma_{ijkl} = < \Psi | a_i^\dagger a_k^\dagger a_l a_j | \Psi >
        Chemist's notation is also implemented
        ..math::
            \Gamma_{ijkl} = < \Psi | a_i^\dagger a_j^\dagger a_k a_l | \Psi >

        Paramaters
        ----------
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            `n`is the `n`th order excitation
        is_chemist_notation : bool
            True if chemist's notation
            False if physicist's notation
            Default is Physicist's notation
        val_threshold : float
            Threshold for truncating the density matrice entries
            Skips all of the Slater determinants whose maximal sum of contributions to density
            matrices is less than threshold value

        Returns
        -------
        one_densities : tuple of np.ndarray
            One electron density matrix
            For spatial and generalized orbitals, 1-tuple of np.ndarray
            For unretricted spin orbitals, 2-tuple of np.ndarray
        two_densities : tuple of np.ndarray
            Two electron density matrix
            For spatial and generalized orbitals, 1-tuple of np.ndarray
            For unrestricted orbitals, 3-tuple of np.ndarray
        """
        return density_matrix(self.sd_coeffs[:, self.dict_exc_index[exc_lvl]].flat, self.civec,
                              self.nspatial, is_chemist_notation=is_chemist_notation,
                              val_threshold=val_threshold, orbtype=self.orbtype)


    ###########
    # Solving #
    ###########
    def compute_ci_matrix(self):
        """ Returns CI Hamiltonian matrix in the Slater determinant basis

        ..math::
            H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

        Returns
        -------
        matrix : np.ndarray(K, K)
        """
        return ci_matrix(self.one_int, self.two_int, self.civec, self.dtype, self.orbtype)


    def generate_civec(self):
        """ Generates Slater determinants

        All orders of excitations given the assigned spin and seniority

        Returns
        -------
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring
        """
        return sd_list(self.nelec, self.nspatial, num_limit=None, exc_orders=None, spin=self.spin,
                       seniority=self.seniority)