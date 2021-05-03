from wfns.wfn.geminal.apsetg4 import BasicAPsetG4


class BasicAPsetG7(BasicAPsetG4):
    def __init__(
        self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None, tol=1e-4
    ):
        super().__init__(nelec, nspin, dtype=dtype, memory=memory, ngem=ngem, orbpairs=orbpairs,
                         params=params, tol=tol, num_matchings=1)
