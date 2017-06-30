"""Hamiltonian object that interacts with the wavefunction.

..math::
    \hat{H} = \sum_{ik} h_{ik} a^\dagger_i a_k
    + \sum_{ijkl} g_{ijkl} a^\dagger_i a^\dagger_j a_k a_l
where :math:`h_{ik}` is the one-electron integral and :math:`g_{ijkl}` is the two-electron integral.

Class
-----
ChemicalHamiltonian(one_int, two_int, orbtype=None, energy_nuc_nuc=None)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from itertools import combinations
from ..backend.integrals import OneElectronIntegrals, TwoElectronIntegrals
from ..backend import slater


class ChemicalHamiltonian(object):
    """Hamiltonian used for a typical chemical system.

    ..math::
        \hat{H} &= \\
        &= \sum_{ij} h_{ij} a^\dagger_i a_j + \sum_{ijkl} g_{ijkl} a^\dagger_i a^\dagger_j a_k a_l\\

    Attributes
    ----------
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used.
    energy_nuc_nuc : float
        Nuclear-nuclear repulsion energy
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

    Properties
    ----------
    dtype
        Data type of the integrals.

    Methods
    -------
    assign_orbtype(self, dtype)
        Assigns the orbital type.
    assign_energy_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    """

    def __init__(self, one_int, two_int, orbtype=None, energy_nuc_nuc=None):
        """Initialize the Hamiltonian.

        Parameters
        ----------
        one_int : np.ndarray(K,K), 1- or 2-tuple np.ndarray(K,K)
            One-electron integrals
            If orbitals are spatial or generalized, then integrals are given as np.ndarray or
            1-tuple of np.ndarray
            If orbitals are unretricted, then integrals are 2-tuple of np.ndarray
        two_int : np.ndarray(K,K,K,K), 1- or 3-tuple np.ndarray(K,K,K,K)
            If orbitals are spatial or generalized, then integrals are given as np.ndarray or
            1-tuple of np.ndarray
            If orbitals are unretricted, then integrals are 3-tuple of np.ndarray
        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbitals used.
            Default is `'restricted'`
        energy_nuc_nuc : {float, None}
            Nuclear nuclear repulsion energy
            Default is `0.0`

        """
        self.assign_orbtype(orbtype)
        self.assign_energy_nuc_nuc(energy_nuc_nuc)
        self.assign_integrals(one_int, two_int)

    @property
    def nspin(self):
        """Return the number of spin orbitals."""
        if self.orbtype in ('restricted', 'unrestricted'):
            return 2*self.one_int.num_orbs
        elif self.orbtype == 'generalized':
            return self.one_int.num_orbs
        else:
            raise NotImplementedError('Unsupported orbital type.')

    @property
    def dtype(self):
        """Return the data type of the integrals."""
        return self.one_int.dtype

    # FIXME: getter/setter is not used b/c assign_integrals is a little complicated.
    def assign_orbtype(self, orbtype=None):
        """Assign the orbital type.

        Parameters
        ----------
        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbitals used.
            Default is `'restricted'`

        Raises
        ------
        ValueError
            If orbtype is not one of ['restricted', 'unrestricted', 'generalized']

        Note
        ----
        Should be executed before assign_integrals.
        """
        if orbtype is None:
            orbtype = 'restricted'
        if orbtype not in ['restricted', 'unrestricted', 'generalized']:
            raise TypeError("Orbital type must be one of 'restricted', 'unrestricted', "
                            "and 'generalized'.")
        self.orbtype = orbtype

    def assign_energy_nuc_nuc(self, energy_nuc_nuc=None):
        """Assigns the nuclear nuclear repulsion.

        Parameters
        ----------
        energy_nuc_nuc : int, float, None
            Nuclear-nuclear repulsion energy
            Default is 0.0

        Raises
        ------
        TypeError
            If energy_nuc_nuc is not int or float or None
        """
        if energy_nuc_nuc is None:
            energy_nuc_nuc = 0.0
        elif isinstance(energy_nuc_nuc, (int, float)):
            energy_nuc_nuc = float(energy_nuc_nuc)
        else:
            raise TypeError('Nuclear-nuclear repulsion must be given as a int, float, or None.')
        self.energy_nuc_nuc = energy_nuc_nuc

    def assign_integrals(self, one_int, two_int):
        """Assign the one- and two-electron integrals.

        Parameters
        ----------
        one_int : {np.ndarray(K,K), 1- or 2-tuple np.ndarray(K,K), OneElectronIntegrals}
            One electron integrals for restricted, unrestricted, or generalized orbitals
            1-tuple for spatial (restricted) and generalized orbitals
            2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
        two_int : {np.ndarray(K,K), 1- or 3-tuple np.ndarray(K,K), TwoElectronIntegrals}
            Two electron integrals for restricted, unrestricted, or generalized orbitals
            In physicist's notation
            1-tuple for spatial (restricted) and generalized orbitals
            3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
            beta-beta-beta-beta components)

        Raises
        ------
        TypeError
            If integrals are not provided as a numpy array or a tuple of numpy arrays
            If integral matrices do not have dtype of `float` or `complex`
            If integral matrices do not have the same dtype
            If the number of one-electron integral matrices is not 1 or 2
            If one-electron integral matrices are not two dimensional
            If one-electron integral matrices are not square
            If one-electron integrals (for the unrestricted orbitals) has different number of alpha
            and beta orbitals
            If the number of two-electron integral matrices is not 1 or 3
            If two-electron integral matrices are not four dimensional
            If two-electron integral matrices does not have same dimensionality in all axis
            If two-electron integrals (for the unrestricted orbitals) has different number of alpha
            and beta orbitals
            If one- and two-electron integrals do not have the same number of orbitals
            If one- and two-electron integrals do not have the same data type
            If one- and two-electron integrals do not correspond to the same orbital type (i.e.
            restricted, unrestricted, generalized).
            If orbital type of the integrals do not match with the given orbital type
        NotImplementedError
            If generalized orbitals and odd number of spin orbitals

        Note
        ----
        Should be executed after assign_orbtype.
        Depending on the orbital type, the form of the integrals can vary:
            Restricted Orbitals
                one_int can be a np.ndarray or a 1-tuple of np.ndarray
                two_int can be a np.ndarray or a 1-tuple of np.ndarray
                orbtype = 'restricted'
            Unrestricted Orbitals
                one_int can be a 2-tuple of np.ndarray
                two_int can be a 3-tuple of np.ndarray
                orbtype = 'unrestricted'
            Generalized Orbitals
                one_int can be a np.ndarray or a 1-tuple of np.ndarray
                two_int can be a np.ndarray or a 1-tuple of np.ndarray
                orbtype = 'generalized'
        """
        one_int = OneElectronIntegrals(one_int)
        two_int = TwoElectronIntegrals(two_int)
        if one_int.num_orbs != two_int.num_orbs:
            raise TypeError('One- and two-electron integrals do not have the same number of '
                            'orbitals.')
        elif one_int.dtype != two_int.dtype:
            raise TypeError('One- and two-electron integrals do not have the same data type.')
        elif one_int.possible_orbtypes != two_int.possible_orbtypes:
            raise TypeError('One- and two-electron integrals do not correspond to the same orbital '
                            'type (i.e. restricted, unrestricted, generalized).')
        elif self.orbtype not in one_int.possible_orbtypes:
            raise TypeError('Orbital type of the integrals do not match with the given orbital '
                            'type, {0}'.format(self.orbtype))
        elif self.orbtype == 'generalized' and one_int.num_orbs % 2 == 1:
            raise NotImplementedError('Odd number of "spin" orbitals will cause problems when '
                                      ' constructing Slater determinants.')

        self.one_int = one_int
        self.two_int = two_int

    def integrate_wfn_sd(self, wfn, sd, deriv=None):
        """Integrates the Hamiltonian with against a wavefunction and Slater determinant.

        ..math::
            \big< \Psi \big| \hat{H} \big| \Phi \big>

        where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
        :math:`\Phi` is the Slater determinant.

        Parameters
        ----------
        wfn : Wavefunction
            Wavefunction against which the Hamiltonian is integrated.
            Needs to have the following in `__dict__`: `get_overlap`.
        sd : int
            Slater Determinant against which the Hamiltonian is integrated.
        deriv : int, None
            Index of the parameter against which the expectation value is derivatized.
            Default is no derivatization

        Returns
        -------
        one_electron : float
            One electron energy
        coulomb : float
            Coulomb energy
        exchange : float
            Exchange energy
        """
        # FIXME: incredibly slow/bad approach
        occ_indices = slater.occ_indices(sd)
        vir_indices = slater.vir_indices(sd, self.nspin)

        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0

        # sum over zeroth order excitation
        coeff = wfn.get_overlap(sd, deriv=deriv)
        for counter, i in enumerate(occ_indices):
            one_electron += coeff * self.one_int.get_value(i, i, self.orbtype)
            for j in occ_indices[counter+1:]:
                coulomb += coeff * self.two_int.get_value(i, j, i, j, self.orbtype)
                exchange -= coeff * self.two_int.get_value(i, j, j, i, self.orbtype)

        # sum over one electron excitation
        for counter, i in enumerate(occ_indices):
            for a in vir_indices:
                coeff = wfn.get_overlap(slater.excite(sd, i, a), deriv=deriv)
                one_electron += coeff * self.one_int.get_value(i, a, self.orbtype)
                for j in occ_indices[:counter] + occ_indices[counter+1:]:
                    coulomb += coeff * self.two_int.get_value(i, j, a, j, self.orbtype)
                    exchange -= coeff * self.two_int.get_value(i, j, j, a, self.orbtype)

        # sum over two electron excitation
        for i, j in combinations(occ_indices, 2):
            for a, b in combinations(vir_indices, 2):
                coeff = wfn.get_overlap(slater.excite(sd, i, j, a, b), deriv=deriv)
                coulomb += coeff * self.two_int.get_value(i, j, a, b, self.orbtype)
                exchange -= coeff * self.two_int.get_value(i, j, b, a, self.orbtype)

        return one_electron, coulomb, exchange

    def integrate_sd_sd(self, sd1, sd2):
        """Integrates the Hamiltonian with against two Slater determinants.

        ..math::
            H_{ij} = \big< \Phi_i \big| \hat{H} \big| \Phi_j \big>

        where :math:`\hat{H}` is the Hamiltonian operator, and :math:`\Phi_1` and :math:`\Phi_2` are
        Slater determinants.

        Parameters
        ----------
        sd1 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater Determinant against which the Hamiltonian is integrated.

        Returns
        -------
        one_electron : float
            One electron energy
        coulomb : float
            Coulomb energy
        exchange : float
            Exchange energy
        """
        # FIXME: incredibly slow/bad approach
        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0

        shared_indices = slater.occ_indices(slater.shared(sd1, sd2))
        diff_sd1, diff_sd2 = slater.diff(sd1, sd2)
        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return 0.0, 0.0, 0.0
        diff_order = len(diff_sd1)

        # moving all the shared orbitals toward one another (in the middle)
        num_transpositions_0 = sum(len([j for j in shared_indices if j < i]) for i in diff_sd1)
        num_transpositions_1 = sum(len([j for j in shared_indices if j < i]) for i in diff_sd2)
        num_transpositions = num_transpositions_0 + num_transpositions_1
        sign = (-1)**num_transpositions

        # two sd's are the same
        if diff_order == 0:
            for count, i in enumerate(shared_indices):
                one_electron += sign * self.one_int.get_value(i, i, self.orbtype)
                for j in shared_indices[count + 1:]:
                    coulomb += sign * self.two_int.get_value(i, j, i, j, self.orbtype)
                    exchange -= sign * self.two_int.get_value(i, j, j, i, self.orbtype)

        # two sd's are different by single excitation
        elif diff_order == 1:
            i, = diff_sd1
            a, = diff_sd2
            one_electron += sign * self.one_int.get_value(i, a, self.orbtype)
            for j in shared_indices:
                coulomb += sign * self.two_int.get_value(i, j, a, j, self.orbtype)
                exchange -= sign * self.two_int.get_value(i, j, j, a, self.orbtype)

        # two sd's are different by double excitation
        elif diff_order == 2:
            i, j = diff_sd1
            a, b = diff_sd2
            coulomb += sign * self.two_int.get_value(i, j, a, b, self.orbtype)
            exchange -= sign * self.two_int.get_value(i, j, b, a, self.orbtype)

        return one_electron, coulomb, exchange

    def orb_rotate_jacobi(self, jacobi_indices, theta):
        """Rotate orbitals using Jacobi matrix.

        Parameters
        ----------
        jacobi_indices : tuple/list of ints
            2-tuple/list of indices of the orbitals that will be rotated
        theta : float
            Angle with which the orbitals are rotated
        """
        self.one_int.rotate_jacobi(jacobi_indices, theta)
        self.two_int.rotate_jacobi(jacobi_indices, theta)
