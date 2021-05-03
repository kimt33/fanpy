"""One reference approximation on geminal wavefunction."""
import numpy as np
from wfns.backend import slater
from wfns.wfn.base import BaseWavefunction


# pylint: disable=E1101
# FIXME: parent was removed to allow easier multiple inheritance. Maybe multiple inheritance should
#        be replaced with a wrapper instead? especially since ordering is necessary w/o splitting
#        the BaseGeminal in two.
class OneRefApprox:
    """One reference approximation to geminal wavefunctions.

    The evaluation of the permanent is drastically cheaper if the dimension of the matrix is
    smaller. In this approximation, a set of orbital pairs is selected to be the reference such that
    the corresponding geminal coefficient submatrix is an identity matrix. Then, permanent
    evaluation can be made drastically cheaper by only considering low order perturbations
    (replacing "occupied" orbital pairs with "virtual" orbital pairs) of the reference orbital
    pairs.

    Properties
    ----------
    params_initial_guess(self)
        Return the template of the parameters of the given wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, ngem=None, orbpairs=None, ref_sd=None,
             ref_orbpairs=None, params=None)
        Initialize the wavefunction.
    assign_ref_sd(self, sd=None)
        Assign the reference Slater determinant.
    assign_ref_orbpairs(self, orbpairs=None)
        Assign the orbital pairs that will be used to construct the reference Slater determinant.
    assign_orbpairs(self, orbpairs=None)
        Assign the orbital pairs that will be used to construct the geminals.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.

    """

    def __init__(
        self, nelec, nspin, ngem=None, orbpairs=None, ref_sd=None, ref_orbpairs=None, params=None
    ):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        ngem : {int, None}
            Number of geminals.
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal.
        ref_sd : {int, None}
            Reference Slater determinant.
        ref_orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct the reference Slater
            determinants.
            Default is the first generated orbital pairing scheme.
        params : np.ndarray
            Geminal coefficient matrix.

        Notes
        -----
        Need to skip over APIG.__init__ because `assign_ref_sd` must come before `assign_params`.

        """
        # pylint: disable=W0233
        BaseWavefunction.__init__(self, nelec, nspin)
        self.assign_ngem(ngem=ngem)
        self.assign_ref_sd(sd=ref_sd)
        self.assign_ref_orbpairs(orbpairs=ref_orbpairs)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)

    @property
    def params_initial_guess(self):
        """Return the template of the parameters of the given wavefunction.

        Since part of the coefficient matrix is constrained, these parts will be removed from the
        coefficient matrix.

        Returns
        -------
        params_initial_guess : np.ndarray(ngem, norbpair)
            Default parameters of the geminal wavefunction.

        """
        params = np.zeros((self.ngem, self.norbpair))
        return params

    def assign_ref_sd(self, sd=None):
        """Assign the reference Slater determinant.

        Parameters
        ----------
        sd : {int, None}
            Slater determinant to use as a reference.
            Default is the HF ground state.

        Raises
        ------
        TypeError
            If given `sd` cannot be turned into a Slater determinant (i.e. not integer or list of
            integers).
        ValueError
            If given `sd` does not have the correct spin.
            If given `sd` does not have the correct seniority.

        Notes
        -----
        This method depends on `nelec`, `nspin`, `spin`, and `seniority`.

        """
        # pylint: disable=C0103
        if sd is None:
            sd = slater.ground(self.nelec, self.nspin)
        sd = slater.internal_sd(sd)
        if slater.total_occ(sd) != self.nelec:
            raise ValueError(
                "Given Slater determinant does not have the correct number of " "electrons"
            )
        if self.spin is not None and slater.get_spin(sd, self.nspatial):
            raise ValueError("Given Slater determinant does not have the correct spin.")
        if self.seniority is not None and slater.get_seniority(sd, self.nspatial):
            raise ValueError("Given Slater determinant does not have the correct seniority.")
        self.ref_sd = sd

    def assign_ref_orbpairs(self, orbpairs=None):
        """Assign the orbital pairs that will be used to construct the reference Slater determinant.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple/list of ints
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Raises
        ------
        TypeError
            If an orbital pair is not given as a list or a tuple.
            If an orbital pair does not contain exactly two elements.
            If an orbital index is not an integer.
        ValueError
            If an orbital pair has the same integer.
            If given orbital pairs do not correspond to the reference Slater determinant.

        """
        if orbpairs is None:
            orbpairs = next(self.generate_possible_orbpairs(self.ref_sd))

        # FIX E: copied from BaseGeminal.assign_orbpair
        dict_reforbpair_ind = {}
        for i, orbpair in enumerate(orbpairs):
            if not isinstance(orbpair, (list, tuple)):
                raise TypeError("Each orbital pair must be a list or a tuple")
            if len(orbpair) != 2:
                raise TypeError("Each orbital pair must contain two elements")
            if not (isinstance(orbpair[0], int) and isinstance(orbpair[1], int)):
                raise TypeError("Each orbital index must be given as an integer")
            if orbpair[0] == orbpair[1]:
                raise ValueError("Orbital pair of the same orbital is invalid")
            if not slater.occ(self.ref_sd, orbpair[0]) or not slater.occ(self.ref_sd, orbpair[1]):
                raise ValueError(
                    "Provided orbital pairing does not match with the reference Slater"
                    " determinant"
                )

            orbpair = tuple(orbpair)
            # sort orbitals within the pair
            if orbpair[0] > orbpair[1]:
                orbpair = orbpair[::-1]

            if orbpair in dict_reforbpair_ind:
                raise ValueError(
                    "The given orbital pairs have multiple entries of {0}" "".format(orbpair)
                )
            dict_reforbpair_ind[orbpair] = i

        self.dict_reforbpair_ind = dict_reforbpair_ind

    def assign_orbpairs(self, orbpairs=None):
        """Assign the orbital pairs that will be used to construct the geminals.

        Since a part of the coefficient matrix is constrained, the column indices that correspond to
        these parts will be ignored. Instead, the column indices of the coefficient matrix after the
        removal of the constrained parts will be used.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple/list of ints
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Raises
        ------
        TypeError
            If `orbpairs` is not an iterable.
            If an orbital pair is not given as a list or a tuple.
            If an orbital pair does not contain exactly two elements.
            If an orbital index is not an integer.
        ValueError
            If an orbital pair has the same integer.
            If an orbital pair occurs more than once.

        Notes
        -----
        Must have `ref_sd` and `nspin` defined for the default option.

        """
        super().assign_orbpairs(orbpairs=orbpairs)
        # removing orbital indices that correspond to the reference Slater determinant
        dict_orbpair_ind = {}
        for orbpair in self.dict_ind_orbpair.values():  # pylint: disable=E0203
            # if orbital pair is occupied in the reference Slater determinant
            dict_orbpair_ind[orbpair] = len(dict_orbpair_ind)

        self.dict_orbpair_ind = dict_orbpair_ind
        self.dict_ind_orbpair = {i: orbpair for orbpair, i in self.dict_orbpair_ind.items()}

    def _olp(self, sd):  # pylint: disable=C0103
        """Calculate the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.

        Returns
        -------
        olp : {float, complex}
            Overlap of the current instance with the given Slater determinant.

        """
        # NOTE: Need to recreate spatial_ref_sd, inds_annihilated and inds_created

        orbs_annihilated, orbs_created = slater.diff_orbs(self.ref_sd, sd)
        inds_annihilated = slater.occ_indices(orbs_annihilated)
        inds_created = slater.occ_indices(orbs_created)

        # FIXME: missing signature. see apig. Not a problem if alpha beta spin pairing
        val = 0.0
        # FIXME: generate_possible_orbpairs only uses dict_orbpair_ind (not dict_reforbpair_ind)
        for c_orbpairs, c_sign in self.generate_possible_orbpairs(inds_created):
            if not c_orbpairs:
                continue
            col_inds = np.array([self.dict_orbpair_ind[orbp] for orbp in c_orbpairs])
            # find orbpairs indices that are annihilated
            for a_orbpairs, a_sign in self.generate_possible_orbpairs(inds_annihilated):
                if not a_orbpairs:
                    continue

                # signature of annihilation corresponds to the permutation that makes it
                # strictly increasing.
                # Since signature for creation is strictly increasing,
                # FIXME: signature is wrong
                row_inds = np.array([self.dict_reforbpair_ind[orbp] for orbp in a_orbpairs])
                val += c_sign * a_sign * self.compute_permanent(col_inds, row_inds=row_inds)
        return val

    # TODO: not implemented
    def _olp_deriv(self, sd, deriv):  # pylint: disable=C0103
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
        deriv : int
            Index of the parameter with respect to which the overlap is derivatized.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        """
        raise NotImplementedError

    # FIXME: allow other pairing schemes.
    # FIXME: too many return statements
    def get_overlap(self, sd, deriv=None):  # pylint: disable=C0103,R1710,R0911
        """Return the overlap of the wavefunction with a Slater determinant.

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : int
            Index of the parameter to derivatize.
            Default does not derivatize.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        Notes
        -----
        Pairing scheme is assumed to be the alpha-beta spin orbital pair of each spatial orbital.
        This code will fail if another pairing scheme is used.

        """
        sd = slater.internal_sd(sd)

        # FIXME: hardcodes alpha-beta pairing scheme
        # cut off beta part (for just the alpha/spatial part)
        spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
        spatial_sd, _ = slater.split_spin(sd, self.nspatial)
        # get indices of the occupied orbitals
        orbs_annihilated, orbs_created = slater.diff_orbs(spatial_ref_sd, spatial_sd)

        # if different number of electrons
        if len(orbs_annihilated) != len(orbs_created):
            return 0.0
        # if different seniority
        if slater.get_seniority(sd, self.nspatial) != 0:
            return 0.0

        # convert to spatial orbitals
        # NOTE: these variables are essentially the same as the output of
        #       generate_possible_orbpairs
        inds_annihilated = np.array(
            [self.dict_reforbpair_ind[(i, i + self.nspatial)] for i in orbs_annihilated]
        )
        inds_created = np.array(
            [self.dict_orbpair_ind[(i, i + self.nspatial)] for i in orbs_created]
        )

        # if no derivatization
        if deriv is None:
            if inds_annihilated.size == inds_created.size == 0:
                return 1.0

            return self._olp(sd)
        # if derivatization
        if isinstance(deriv, (int, np.int64)):
            if deriv >= self.nparams:
                return 0.0
            if inds_annihilated.size == inds_created.size == 0:
                return 0.0

            return self._olp_deriv(sd, deriv)
