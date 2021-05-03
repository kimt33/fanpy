"""Wavefunction using Keras NN.

keras = 2.2.4
tensorflow = 1.14.0
"""
from tensorflow.keras import backend, layers, activations, models
import numpy as np
import wfns.backend.slater as slater
from wfns.wfn.base import BaseWavefunction
import cachetools


class KerasNetwork(BaseWavefunction):
    r"""Base wavefunction class.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.

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
    template_params : np.ndarray
        Default parameters of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_dtype(self, dtype)
        Assign the data type of the parameters.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_params(self, params)
        Assign parameters of the wavefunction.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.

    """

    # pylint: disable=W0223
    def __init__(self, nelec, nspin, model=None, params=None, dtype=None, memory=None, num_layers=2):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        mode : {keras.Model, None}
            Model instance from keras.
            Default is 2 layers.
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type.
            Default is `np.float64`.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).

        """
        super().__init__(nelec, nspin, dtype=dtype, memory=memory)
        self.num_layers = num_layers
        self.assign_model(model=model)
        self._template_params = None
        self.load_cache()
        self.assign_params(params=params)

    def assign_dtype(self, dtype=None):
        """Assign the data type of the parameters.

        Parameters
        ----------
        dtype : {float, complex, np.float64, np.complex128}
            Numpy data type.
            If None then set to np.float64.

        Raises
        ------
        TypeError
            If dtype is not one of float, complex, np.float64, np.complex128.
        ValueError
            If dtype is not np.float64.

        """
        super().assign_dtype(dtype)
        if self.dtype != np.float64:
            raise ValueError("Given data type must be a np.float64.")
        backend.set_floatx('float64')
        #backend.common._FLOATX = "float64"  # pylint: disable=W0212

    def assign_model(self, model=None):
        """Assign the Keras model used to represent the neural network.

        Parameters
        ----------
        model : {keras.engine.training.Model, None}
            Keras Model instance.
            Default is a neural network with two hidden layers with ReLU activations. The number of
            hidden units in each layer is the number of spin orbitals.

        Raises
        ------
        TypeError
            If the given model is not an instance of keras.engine.training.Model.
        ValueError
            If the number of "types" of input variables is not one.
            If the number of input variables is not the number of spin orbitals.
            If the number of "types" of output variables is not one.
            If the number of output variables is not one.

        """
        if model is None:
            input_layer = layers.Input(shape=(self.nspin, ))
            hidden_layer = input_layer
            # model = keras.engine.sequential.Sequential()
            for _ in range(self.num_layers - 1):
                # model.add(
                #     keras.layers.core.Dense(
                #         self.nspin,
                #         activation=keras.activations.tanh,
                #         input_dim=self.nspin,
                #         use_bias=False,
                #     )
                # )
                hidden_layer = layers.Dense(
                    self.nspin, activation=activations.tanh, use_bias=False
                )(hidden_layer)
            # model.add(
            #     keras.layers.core.Dense(
            #         self.nelec, activation=keras.activations.tanh, input_dim=self.nspin, use_bias=False
            #     )
            # )
            output_layer = []
            for _ in range(self.nelec):
                output_layer.append(
                    layers.Dense(
                        1, activation=activations.tanh, use_bias=False
                    )(hidden_layer)
                )
            model = models.Model(inputs=input_layer, outputs=output_layer)

        # if not isinstance(model, keras.engine.training.Model):
        #    raise TypeError("Given model must be an instance of keras.engine.network.Network.")
        if len(model.inputs) != 1:
            raise ValueError(
                "Given model must only have one set of inputs (for the occupations of "
                "the Slater determinant)."
            )
        if model.inputs[0].shape[1] != self.nspin:
            raise ValueError(
                "Given model must have exactly the same number of input nodes as the "
                "number of spin orbitals."
            )
        # if len(model.outputs) != 1:
        #     raise ValueError(
        #         "Given model must only have one set of outputs (for the overlap of "
        #         "the Slater determinant)."
        #     )
        # if model.outputs[0].shape[1] != 1:
        #     raise ValueError("Given model must have exactly one output.")

        # compile model because some of the methods/attributes do not exist until compilation
        # (e.g. get_gradient)
        # NOTE: Keras has a built in method for gradient but it includes the loss function. To make
        # things easier, we modify the loss function so that it does not do anything (i.e. identify
        # function)
        def loss(y_true, y_pred):
            """Loss function used to hack in objective into Keras."""
            return backend.sum(y_true - y_pred)

        model.compile(loss=loss, optimizer="sgd")

        self.model = model

    @property
    def nparams(self):
        """Return the number of wavefunction parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return self.model.count_params()
        # return self.model.count_params() + 1

    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        Notes
        -----
        Instance must have attribut `model`.

        """
        return (self.nparams,)

    @property
    def template_params(self):
        """Return the template of the parameters of the given wavefunction.

        Returns
        -------
        template_params : np.ndarray
            Default parameters of the wavefunction.

        Notes
        -----
        May depend on params_shape and other attributes/properties.

        """
        return self._template_params

    # FIXME: not a very robust way of building an initial guess. It is not very good and requires
    # specific network structures.
    def assign_template_params(self):
        r"""Assign the intial guess for the HF ground state wavefunction.

        Since the template parameters are calculated/approximated, they are computed and stored away
        rather than generating each one on the fly.

        Raises
        ------
        ValueError
            If any of the layers of the model has more than one type of weights. For example, bias
            is not allowed.
            If the number of units `K` in the final hidden layer is greater than
            :math:`1 + (K-N)N + \binom{K-N}{2} \binom{N}{2}`.

        Notes
        -----
        The template parameters can only be created for networks without bias and sufficiently large
        final hidden layer. Additionally, the produced parameters may not be a good initial guess
        for the HF ground state.

        """
        params = []
        scale = 1 / np.tanh(1)
        if self.num_layers > 1:
            params.extend(np.eye(*self.model.layers[1].weights[0].shape).flatten())
            for layer in self.model.layers[2:-self.nelec]:
                params.extend(np.eye(*layer.weights[0].shape).flatten() * scale)

        # FIXME: hardcoded structure
        output_weights = np.zeros((self.nspin, self.nelec))
        npair = self.nelec // 2
        output_weights[np.arange(npair), np.arange(npair)] = 1
        output_weights[
            np.arange(self.nspatial, self.nspatial + npair), np.arange(npair, self.nelec)
        ] = 1
        params.extend(output_weights.T.flatten() * scale)

        self.output_scale = scale
        # self.params[-1] = scale

        self._template_params = np.array(params, dtype=self.dtype)

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
        add_noise : bool
            Flag to add noise to the given parameters.

        Raises
        ------
        TypeError
            If `params` is not a numpy array.
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`.
            If `params` has complex data type and wavefunction has float data type.
        ValueError
            If `params` does not have the same shape as the template_params.

        Notes
        -----
        Depends on dtype, template_params, and nparams.

        """
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self.template_params

        # store parameters
        super().assign_params(params=params, add_noise=add_noise)

        # divide parameters into a list of two dimensional numpy arrays
        weights = []
        counter = 0
        for var_weights in self.model.weights:
            next_counter = counter + np.prod(var_weights.shape)
            var_params = self.params[counter:next_counter].reshape(var_weights.shape)
            weights.append(var_params)
            counter = next_counter
        # change weights of model
        self.model.set_weights(weights)
        self.clear_cache()

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

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

        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally.

        Notes
        -----
        Overlaps and their derivatives are not cached.

        """
        # if no derivatization
        if deriv is None:
            return self._olp(sd)
        return self._olp_deriv(sd)

    @cachetools.cachedmethod(cache=lambda obj: obj._cache_fns["overlap"])
    def _olp(self, sd):
        occ_vector = np.zeros(self.nspin)
        occ_vector[np.array(slater.occ_indices(sd))] = 1
        occ_vector = occ_vector[None, :]

        scale = self.output_scale
        # scale = self.params[-1]
        vals = np.hstack(self.model.predict(occ_vector))
        return np.prod(vals * scale)
        # return np.prod(vals * scale, axis=1)

    @cachetools.cachedmethod(cache=lambda obj: obj._cache_fns["overlap derivative"])
    def _olp_deriv(self, sd):
        occ_vector = np.zeros(self.nspin)
        occ_vector[np.array(slater.occ_indices(sd))] = 1
        occ_vector = occ_vector[None, :]

        scale = self.output_scale
        # scale = self.params[-1]
        vals = np.hstack(self.model.predict(occ_vector)) * scale
        grads = backend.function(
            self.model.inputs,
            [
                j
                for i in range(self.nelec)
                for j in
                self.model.optimizer.get_gradients(
                    self.model.outputs[i],
                    self.model.weights[:-self.nelec] + [self.model.weights[-self.nelec + i]],
                )
            ],
        )([occ_vector])

        output = np.zeros(self.nparams)
        for i in range(self.nelec):
            grads_i = grads[self.num_layers * i : self.num_layers * (i + 1)]
            # FIXME: hardcoded structure
            output_layer_size = grads_i[-1].size

            temp = np.hstack([j.flatten() * scale for j in grads_i])
            temp *= np.prod(vals[:, :i])
            temp *= np.prod(vals[:, i+1:])
            output[:-self.nelec * output_layer_size] += temp[:-output_layer_size]
            # output[:-self.nelec * output_layer_size - 1] += temp[:-output_layer_size]
            if i + 1 < self.nelec:
                output[
                    (-self.nelec + i) * output_layer_size: (-self.nelec + i + 1) * output_layer_size
                    # (-self.nelec + i) * output_layer_size - 1:
                    # (-self.nelec + i + 1) * output_layer_size - 1
                ] += temp[-output_layer_size:]
            else:
                output[(-self.nelec + i) * output_layer_size:] += temp[-output_layer_size:]
                # output[(-self.nelec + i) * output_layer_size - 1: -1] += temp[-output_layer_size:]
        # output[-1] = np.prod(vals) * self.nelec / scale

        return output

    def normalize(self, pspace):
        norm = sum(self.get_overlap(sd)**2 for sd in pspace)
        # FIXME: hard coded structure
        self.output_scale *= norm ** (-0.5 / self.nelec)
        # self.params[-1] = norm ** (-0.5 / self.nelec)
        self.clear_cache()
