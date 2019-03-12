import numpy
import scipy.linalg

class FieldConfig(object):
    """Object for managing stored auxilliary field.

    Parameters
    ----------
    nfields : int
        Number of fields to store for each back propagation step.
    nprop_tot : int
        Total number of propagators to store for back propagation + itcf.
    nbp : int
        Number of back propagation steps.
    dtype : type
        Field configuration type.
    """
    def __init__(self, nfields, nprop_tot, nbp, dtype):
        self.configs = numpy.zeros(shape=(nprop_tot, nfields), dtype=dtype)
        self.cos_fac = numpy.zeros(shape=(nprop_tot, 1), dtype=float)
        self.weight_fac = numpy.zeros(shape=(nprop_tot, 1), dtype=complex)
        self.tot_wfac = 1.0 + 0j
        self.step = 0
        # need to account for first iteration and how we iterate
        self.block = -1
        self.ib = 0
        self.nfields = nfields
        self.nbp = nbp
        self.nprop_tot = nprop_tot
        self.nblock = nprop_tot // nbp

    def push(self, config):
        """Add field configuration to buffer.

        Parameters
        ----------
        config : int
            Auxilliary field configuration.
        """
        self.configs[self.step, self.ib] = config
        self.ib = (self.ib + 1) % self.nfields
        # Completed field configuration for this walker?
        if self.ib == 0:
            self.step = (self.step + 1) % self.nprop_tot
            # Completed this block of back propagation steps?
            if self.step % self.nbp == 0:
                self.block = (self.block + 1) % self.nblock

    def update(self, config, wfac):
        """Add full field configuration for walker to buffer.

        Parameters
        ----------
        config : :class:`numpy.ndarray`
            Auxilliary field configuration.
        cfac : float
            Cosine factor if using phaseless approximation.
        wfac : complex
            Weight factor to restore full walker weight following phaseless
            approximation.
        """
        self.configs[self.step] = config
        self.weight_fac[self.step] = wfac[0]
        self.cos_fac[self.step] = wfac[1]
        try:
            self.tot_wfac *= wfac/cfac
        except ZeroDivisionError:
            self.tot_wfac = 0.0
        # Completed field configuration for this walker?
        self.step = (self.step + 1) % self.nprop_tot
        # Completed this block of back propagation steps?
        if self.step % self.nbp == 0:
            self.block = (self.block + 1) % self.nblock

    def get_block(self):
        """Return a view to current block for back propagation."""
        start = self.block * self.nbp
        end = (self.block + 1) * self.nbp
        return (self.configs[start:end], self.cos_fac[start:end],
                self.weight_fac[start:end])

    def get_superblock(self):
        """Return a view to current super block for ITCF."""
        end = self.nprop_tot - self.nbp
        return (self.configs[:end], self.cos_fac[:end], self.weight_fac[:end])

    def get_wfac(self):
        weight_fac = [1,1]
        for c, w in zip(self.cosine_fac, self.weight_fac):
            weight_fac[0] *= w
            weight_fac[1] *= c
        return weight_fac

    def get_buffer(self):
        buff = {
            'configs': self.configs,
            'cos_fac': self.cos_fac,
            'weight_fac': self.weight_fac
        }
        return buff

    def set_buffer(self, buff):
        self.configs = numpy.copy(buff['configs'])
        self.weight_fac = numpy.copy(buff['weight_fac'])
        self.cos_fac = numpy.copy(buff['cos_fac'])

class PropagatorStack:
    def __init__(self, stack_size, ntime_slices, nbasis, dtype, BT=None, BTinv=None,
                 diagonal=False):
        self.time_slice = 0
        self.stack_size = stack_size
        self.ntime_slices = ntime_slices
        self.nbins = ntime_slices // self.stack_size
        self.diagonal_trial = diagonal
        self.reortho = 1

        if self.nbins * self.stack_size < self.ntime_slices:
            print("stack_size must divide the total path length")
            assert(self.nbins * self.stack_size == self.ntime_slices)

        self.nbasis = nbasis
        self.dtype = dtype
        self.BT = BT
        self.BTinv = BTinv
        self.counter = 0
        self.block = 0
        self.wfac = numpy.array([1.0,1.0], dtype=numpy.complex128)
        I = numpy.identity(nbasis, dtype=dtype)
        self.stack = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                 dtype=dtype)
        self.left = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                dtype=dtype)
        self.right = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                 dtype=dtype)
        # set all entries to be the identity matrix
        self.reset()

    def get(self, ix):
        return self.stack[ix]

    def get_buffer(self):
        buff = {
            'left': self.left,
            'right': self.right,
            'stack': self.stack,
            'wfac': self.wfac,
        }
        return buff

    def set_buffer(self, buff):
        self.stack = numpy.copy(buff['stack'])
        self.left = numpy.copy(buff['left'])
        self.right = numpy.copy(buff['right'])
        self.wfac = buff['wfac']

    def set_all(self, BT):
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        if self.diagonal_trial:
            for i in range(0, self.ntime_slices):
                ix = i // self.stack_size # bin index
                # Commenting out these two. It is only useful for Hubbard
                self.left[ix,0] = numpy.diag(numpy.multiply(BT[0].diagonal(),self.left[ix,0].diagonal()))
                self.left[ix,1] = numpy.diag(numpy.multiply(BT[1].diagonal(),self.left[ix,1].diagonal()))
                self.stack[ix,0] = self.left[ix,0].copy()
                self.stack[ix,1] = self.left[ix,1].copy()
        else:
            for i in range(0, self.ntime_slices):
                ix = i // self.stack_size # bin index
                self.left[ix,0] = numpy.dot(BT[0],self.left[ix,0])
                self.left[ix,1] = numpy.dot(BT[1],self.left[ix,1])
                self.stack[ix,0] = self.left[ix,0].copy()
                self.stack[ix,1] = self.left[ix,1].copy()


    def reset(self):
        self.time_slice = 0
        self.block = 0
        self.wfac = numpy.array([1.0,1.0], dtype=numpy.complex128)
        for i in range(0, self.nbins):
            self.stack[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.stack[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.right[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.right[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.left[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.left[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)

    def update(self, B, wfac=1.0):
        if self.counter == 0:
            self.stack[self.block,0] = numpy.identity(B.shape[-1], dtype=B.dtype)
            self.stack[self.block,1] = numpy.identity(B.shape[-1], dtype=B.dtype)
        self.stack[self.block,0] = B[0].dot(self.stack[self.block,0])
        self.stack[self.block,1] = B[1].dot(self.stack[self.block,1])
        self.wfac *= wfac
        self.time_slice = self.time_slice + 1
        self.block = self.time_slice // self.stack_size
        self.counter = (self.counter + 1) % self.stack_size

    def update_new(self, B):
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        if self.counter == 0:
            self.right[self.block,0] = numpy.identity(B.shape[-1], dtype=B.dtype)
            self.right[self.block,1] = numpy.identity(B.shape[-1], dtype=B.dtype)

        if self.diagonal_trial:
            self.left[self.block,0] = numpy.diag(numpy.multiply(self.left[self.block,0].diagonal(),self.BTinv[0].diagonal()))
            self.left[self.block,1] = numpy.diag(numpy.multiply(self.left[self.block,1].diagonal(),self.BTinv[1].diagonal()))
        else:
            self.left[self.block,0] = self.left[self.block,0].dot(self.BTinv[0])
            self.left[self.block,1] = self.left[self.block,1].dot(self.BTinv[1])

        self.right[self.block,0] = B[0].dot(self.right[self.block,0])
        self.right[self.block,1] = B[1].dot(self.right[self.block,1])


        if self.diagonal_trial:
            self.stack[self.block,0] = numpy.einsum('ii,ij->ij',self.left[self.block,0],self.right[self.block,0])
            self.stack[self.block,1] = numpy.einsum('ii,ij->ij',self.left[self.block,1],self.right[self.block,1])
        else:
            self.stack[self.block,0] = self.left[self.block,0].dot(self.right[self.block,0])
            self.stack[self.block,1] = self.left[self.block,1].dot(self.right[self.block,1])

        self.time_slice = self.time_slice + 1 # Count the time slice
        self.block = self.time_slice // self.stack_size # move to the next block if necessary
        self.counter = (self.counter + 1) % self.stack_size # Counting within a stack

    def get_wfac(self):
        return self.wfac
