"""Driver to perform AFQMC calculation"""
import sys
import json
import time
import numpy
import warnings
import uuid
from math import exp
import copy
import h5py
from pauxy.estimators.handler import Estimators
from pauxy.qmc.options import QMCOpts
from pauxy.qmc.utils import set_rng_seed
from pauxy.systems.utils import get_system
from pauxy.thermal_propagation.utils import get_propagator
from pauxy.trial_density_matrices.utils import get_trial_density_matrices
from pauxy.utils.misc import get_git_revision_hash
from pauxy.utils.io import to_json, get_input_value
from pauxy.walkers.handler import Walkers

class ThermalAFQMC(object):
    """AFQMC driver.

    Non-zero temperature AFQMC using open ended random walk.

    Parameters
    ----------
    model : dict
        Input parameters for model system.
    qmc_opts : dict
        Input options relating to qmc parameters.
    estimates : dict
        Input options relating to what estimator to calculate.
    trial : dict
        Input options relating to trial wavefunction.
    propagator : dict
        Input options relating to propagator.
    parallel : bool
        If true we are running in parallel.
    verbose : bool
        If true we print out additional setup information.

    Attributes
    ----------
    uuid : string
        Simulation state uuid.
    sha1 : string
        Git hash.
    seed : int
        RNG seed. This is set during initialisation in calc.
    root : bool
        If true we are on the root / master processor.
    nprocs : int
        Number of processors.
    rank : int
        Processor id.
    cplx : bool
        If true then most numpy arrays are complex valued.
    init_time : float
        Calculation initialisation (cpu) time.
    init_time : float
        Human readable initialisation time.
    system : system object.
        Container for model input options.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    propagators : :class:`pauxy.propagation.Projectors` object
        Container for system specific propagation routines.
    estimators : :class:`pauxy.estimators.Estimators` object
        Estimator handler.
    walk : :class:`pauxy.walkers.Walkers` object
        Stores walkers which sample the partition function.
    """

    def __init__(self, comm, options=None, system=None,
                 trial=None, parallel=False, verbose=None):
        if verbose is not None:
            self.verbosity = verbose
            verbose = verbose > 0
        qmc_opts = get_input_value(options, 'qmc', default={},
                                   alias=['qmc_options'],
                                   verbose=self.verbosity>1)
        if qmc_opts.get('beta') is None:
            print("Shouldn't call ThermalAFQMC without specifying beta")
            exit()
        # 1. Environment attributes
        if verbose is not None:
            self.verbosity = verbose
            verbose = verbose > 0
        if comm.rank == 0:
            self.uuid = str(uuid.uuid1())
            self.sha1 = get_git_revision_hash()
        # Hack - this is modified later if running in parallel on
        # initialisation.
        self.root = comm.rank == 0
        self.nprocs = comm.size
        self.rank = comm.rank
        self._init_time = time.time()
        self.run_time = time.asctime(),
        # 2. Calculation objects.
        sys_opts = options.get('system')
        if system is not None:
            self.system = system
        else:
            sys_opts = get_input_value(options, 'system', default={},
                                       alias=['model'],
                                       verbose=self.verbosity>1)
            sys_opts['thermal'] = True
            self.system = get_system(sys_opts=sys_opts, verbose=verbose)
        self.qmc = QMCOpts(qmc_opts, self.system, verbose)
        self.qmc.rng_seed = set_rng_seed(self.qmc.rng_seed, comm)
        self.qmc.ntime_slices = int(round(self.qmc.beta/self.qmc.dt))
        # Overide whatever's in the input file due to structure of FT algorithm.
        self.qmc.nmeasure = 1
        if verbose:
            print("# Number of time slices = %i"%self.qmc.ntime_slices)
        self.cplx = True
        if trial is not None:
            self.trial = trial
        else:
            trial_opts = get_input_value(options, 'trial', default={},
                                         alias=['trial_density'],
                                         verbose=self.verbosity>1)
            self.trial = get_trial_density_matrices(comm, trial_opts,
                                                    self.system, self.cplx,
                                                    self.qmc.beta,
                                                    self.qmc.dt, verbose)

        prop_opts = get_input_value(options, 'propagator', default={},
                                    verbose=self.verbosity>1)
        self.propagators = get_propagator(prop_opts, self.qmc, self.system,
                                          self.trial, verbose)

        self.tsetup = time.time() - self._init_time
        est_opts = get_input_value(options, 'estimators', default={},
                                   alias=['estimates'],
                                   verbose=self.verbosity>1)
        self.estimators = (
            Estimators(est_opts, self.root, self.qmc, self.system,
                       self.trial, self.propagators.BT_BP, verbose)
        )
        self.qmc.ntot_walkers = self.qmc.nwalkers
        # Number of walkers per core/rank.
        self.qmc.nwalkers = int(self.qmc.nwalkers/comm.size)
        # Total number of walkers.
        self.qmc.ntot_walkers = self.qmc.nwalkers * self.nprocs
        if self.qmc.nwalkers == 0:
            if comm.rank == 0:
                print("# WARNING: Not enough walkers for selected core count."
                      "There must be at least one walker per core set in the "
                      "input file. Setting one walker per core.")
            afqmc.qmc.nwalkers = 1
        wlk_opts = get_input_value(options, 'walkers', default={},
                                   alias=['walker', 'walker_opts'],
                                   verbose=self.verbosity>1)
        self.walk = Walkers(wlk_opts, self.system, self.trial,
                           self.qmc, verbose)
        # stabilization frequency might be updated due to wrong user input
        if self.qmc.nstblz != self.propagators.nstblz:
            self.propagators.nstblz = self.qmc.nstblz
        if comm.rank == 0:
            json_string = to_json(self)
            self.estimators.json_string = json_string
            self.estimators.dump_metadata()
            self.estimators.estimators['mixed'].print_key()
            self.estimators.estimators['mixed'].print_header()

    def run(self, walk=None, comm=None, verbose=None):
        """Perform AFQMC simulation on state object using open-ended random walk.

        Parameters
        ----------
        state : :class:`pauxy.state.State` object
            Model and qmc parameters.
        walk: :class:`pauxy.walker.Walkers` object
            Initial wavefunction / distribution of walkers.
        comm : MPI communicator
        """
        if walk is not None:
            self.walk = walk
        self.setup_timers()
        (E_T, ke, pe) = self.walk.walkers[0].local_energy(self.system)
        # Calculate estimates for initial distribution of walkers.
        self.estimators.estimators['mixed'].update(self.system, self.qmc,
                                                   self.trial, self.walk, 0,
                                                   self.propagators.free_projection)
        # Print out zeroth step for convenience.
        self.estimators.estimators['mixed'].print_step(comm, self.nprocs, 0, 1)
        if comm.rank == 0:
            eshift = self.propagators.estimate_eshift(self.walk.walkers[0])
        else:
            eshift = 0
        eshift0 = comm.bcast(eshift, root=0)
        eshift = eshift0

        for step in range(1, self.qmc.nsteps + 1):
            start_path = time.time()
            for ts in range(0, self.qmc.ntime_slices):
                if self.verbosity >= 2 and comm.rank == 0:
                    print(" # Timeslice %d of %d."%(ts, self.qmc.ntime_slices))
                start = time.time()
                eloc = 0.0
                weight = 0.0
                for w in self.walk.walkers:
                    self.propagators.propagate_walker(self.system, w,
                                                      ts, eshift)
                    if (abs(w.weight) > w.total_weight * 0.10) and ts > 0:
                        w.weight = w.total_weight * 0.10
                self.tprop += time.time() - start
                start = time.time()
                if ts % self.qmc.npop_control == 0 and ts != 0:
                    self.walk.pop_control(comm)
                if ts % 1 == 0:
                    wnew = self.walk.walkers[0].total_weight
                    wold = self.walk.walkers[0].old_total_weight
                    eshift = -self.qmc.dt*numpy.log(wnew/wold)
                self.tpopc += time.time() - start
            self.tpath += time.time() - start_path
            start = time.time()
            self.estimators.update(self.system, self.qmc,
                                   self.trial, self.walk, step,
                                   self.propagators.free_projection)
            self.testim += time.time() - start
            self.estimators.print_step(comm, self.nprocs, step, 1,
                                       self.propagators.free_projection)
            self.walk.reset(self.trial)
            eshift = eshift0

    def finalise(self, verbose):
        """Tidy up.

        Parameters
        ----------
        verbose : bool
            If true print out some information to stdout.
        """
        if self.root:
            if self.estimators.back_propagation:
                self.estimators.h5f.close()
            if verbose:
                print("# End Time: %s" % time.asctime())
                print("# Running time : %.6f seconds" %
                      (time.time() - self._init_time))
                print("# Timing breakdown (per processor, per path/slice):")
                print("# - Setup: %f s"%self.tsetup)
                nsteps = self.qmc.nsteps
                nslice = nsteps * self.qmc.ntime_slices
                npcon = nslice // self.qmc.npop_control
                print("# - Path update: %f s"%(self.tpath/nsteps))
                print("# - Propagation: %f s"%(self.tprop/nslice))
                print("# - Estimators: %f s"%(self.testim/nsteps))
                print("# - Population control: %f s"%(self.tpopc/npcon))

    def determine_dtype(self, propagator, system):
        """Determine dtype for trial wavefunction and walkers.

        Parameters
        ----------
        propagator : dict
            Propagator input options.
        system : object
            System object.
        """
        hs_type = propagator.get('hubbard_stratonovich', 'discrete')
        continuous = 'continuous' in hs_type
        twist = system.ktwist.all() is not None
        return continuous or twist

    def setup_timers(self):
        self.tpath = 0
        self.tprop = 0
        self.testim = 0
        self.tpopc = 0
