"""Matrix Product State wavefunction with tanh activation."""
import numpy as np
from wfns.wfn.network.mps import MatrixProductState


class MPSTanh(MatrixProductState):
    """Matrix Product State wavefunction that applies tanh activation after matrix multiplication.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    params : np.ndarray
        Parameters of the wavefunction.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    param_shape : tuple of int
        Shape of the parameters.
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.
    params_initial_guess : np.ndarray
        Default parameters of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_params(self, params)
        Assign parameters of the wavefunction.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.
    assign_dimension(self, dimension=None)
        Assign the dimension of the matrices.
    get_occupation_indices(self, sd) : np.ndarray
        Return the occupation vector of the Slater determinant in the format used in MPS.
    get_matrix_shape(self, index) : 3-tuple of int
        Get the shape of a matrix.
    get_matrix_indices(self, index) : 2-tuple of int
        Get the start and end indices of a matrix.
    get_matrix(self, index) : np.ndarray
        Get the matrix that correspond to the spatial orbital of the given index.
    decompose_index(self, param_index) : 4-tuple of int
        Return the indices of the spatial orbital, occupation, row and column indices.

    """

    def _olp(self, sd):
        """Calculate the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.

        Returns
        -------
        olp : {float, complex}
            Overlap of the current instance with the given Slater determinant.

        """
        occ_indices = self.get_occupation_indices(sd)

        temp_matrix = self.get_matrix(0)[occ_indices[0], :, :]
        for i in range(1, occ_indices.size):
            temp_matrix = temp_matrix.dot(self.get_matrix(i)[occ_indices[i], :, :])
            temp_matrix = np.tanh(temp_matrix)
        return temp_matrix.item()

    def _olp_deriv(self, sd, deriv):
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.
        deriv : int
            Index of the parameter with respect to which the overlap is derivatized.
            Assumed to correspond to the matrix elements that correspond to the given Slater
            determinant.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        """
        deriv_matrix, _, deriv_row, deriv_col = self.decompose_index(deriv)

        occ_indices = self.get_occupation_indices(sd)

        left_temp, right_temp = 1.0, 1.0

        temp_matrix = self.get_matrix(0)[occ_indices[0], :, :]
        for i in range(1, deriv_matrix - 1):
            temp_matrix = temp_matrix.dot(self.get_matrix(i)[occ_indices[i], :, :])
            temp_matrix = np.tanh(temp_matrix)

        der_left = temp_matrix[:, deriv_row]
        (1 - np.tanh(temp_matrix)**2)[:, deriv_row]

        for i in range(1, deriv_matrix - 1):
            pass
        # if selected matrix is not the left most matrix
        if deriv_matrix > 0:
            left_temp = self.get_matrix(0)[occ_indices[0], :, :]
            for i in range(1, deriv_matrix):
                left_temp = left_temp.dot(self.get_matrix(i)[occ_indices[i], :, :])
                left_temp[left_temp < 0] = 0
            left_temp = left_temp[:, deriv_row]
        if left_temp == 0:
            return 0.0

        # if selected matrix is not the right most matrix
        if deriv_matrix < occ_indices.size - 1:
            # index of the matrix to the right of the given matrix
            index_right = deriv_matrix + 1

            # multiply matrix towards the right (from the derivatized matrix)
            right_temp = self.get_matrix(index_right)[occ_indices[index_right], deriv_col, :]
            for i in range(index_right + 1, occ_indices.size):
                right_temp = right_temp.dot(self.get_matrix(i)[occ_indices[i], :, :])

        return (left_temp * right_temp).item()
