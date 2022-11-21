import subprocess
import shlex

import numpy as np

from adder.type_checker import check_type, check_greater_than


class Depletion(object):
    """The class which contains the data and methods needed for
    interfacing with the depletion solver.

    This class should be extended for each and every depletion solver
    for which support is included in ADDER.

    Parameters
    ----------
    exec_cmd : str
        The command needed to run the depletion solver
    num_threads : int
        Number of shared memory threads to utilize
    num_procs : int
        Number of distributed memory instances to utilize
    chunksize : int
        Number of depleting materials to load at a time per thread

    Attributes
    ----------
    solver_type : str
        A shorthand for the solver represented by this instance
    exec_cmd : str
        The command needed to run the depletion solver, including any
        MPI commands
    num_threads : int
        Number of shared memory threads to utilize
    num_procs : int
        Number of distributed memory instances to utilize
    chunksize : int
        Number of depleting materials to load at a time per thread
    decay_data : various
        Pre-computed decay library/matrix data, with a specific type depending
        on the solver.

    """

    def __init__(self, exec_cmd, num_threads, num_procs, chunksize):
        self.exec_cmd = exec_cmd
        self.num_threads = num_threads
        self.num_procs = num_procs
        self.chunksize = chunksize
        self.decay_data = None

        # Instead of logging from this class, we just build and store the
        # log messages to write and store in self._logs
        self.logs = []
        msg = "Initialized Depletion Solver, {}".format(self.solver)
        self.logs.append(("info", msg, 2))

        if exec_cmd != "":
            msg = "Depletion Command: {}".format(self.exec_cmd)
            self.logs.append(("info", msg, 4))

            # Input processing only allows exec_cmds to be passed for solvers
            # with an exec to call, so need for an if-block here.
            try:
                cmd = shlex.split("which {}".format(self.exec_cmd))
                status = subprocess.run(cmd, check=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, shell=False)
                msg = "Depletion Solver Location: {}".format(
                    status.stdout.strip())
                self.logs.append(("info_file", msg, None))
            except subprocess.CalledProcessError:
                msg = "Could Not Find Depletion Solver!"
                self.logs.append(("error", msg, None))

    @property
    def solver(self):
        return "test"

    @property
    def exec_cmd(self):
        return self._exec_cmd

    @exec_cmd.setter
    def exec_cmd(self, exec_cmd):
        check_type("exec_cmd", exec_cmd, str)
        self._exec_cmd = exec_cmd

    @property
    def num_threads(self):
        return self._num_threads

    @num_threads.setter
    def num_threads(self, num_threads):
        check_type("num_threads", num_threads, int)
        check_greater_than("num_threads", num_threads, 0)
        self._num_threads = num_threads

    @property
    def num_procs(self):
        return self._num_procs

    @num_procs.setter
    def num_procs(self, num_procs):
        check_type("num_procs", num_procs, int)
        check_greater_than("num_procs", num_procs, 0)
        self._num_procs = num_procs

    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    def chunksize(self, chunksize):
        check_type("chunksize", chunksize, int)
        check_greater_than("chunksize", chunksize, 0)
        self._chunksize = chunksize

    def compute_decay(self, depl_lib):
        """Computes the decay library. For ORIGEN this is the lib string, for
        CRAM this is the decay matrix"""
        pass

    def compute_library(self, depl_lib, flux):
        """Computes and returns the total library,
        decay + flux-induces reactions. For ORIGEN this is a lib string,
        for CRAM this is the sparse matrix"""
        pass

    def init_h5_file(self, h5_file):
        """Initializes the HDF5 file so subsequent steps can be written
        as they are completed.
        """

        # Store the runtime options as root attributes
        h5_file.attrs["num_depletion_threads"] = self.num_threads
        h5_file.attrs["depletion_chunksize"] = self.num_threads
        h5_file.attrs["depletion_solver"] = np.string_(self.solver)
        h5_file.attrs["depletion_exec"] = np.string_(self.exec_cmd)

    def execute(self, material, A_matrix, iso_indices, inv_iso_indices,
                duration, depletion_step, num_substeps, is_serial=True):
        """Execute the Depletion solver on each of the materials provided.

        Parameters
        ----------
        material : adder.Material
            The problem material
        A_matrix : scipy.csr_matrix or str
            The depletion matrix from the applicable depletion library, or,
            the ORIGEN file ready to be written to disk
        iso_indices : dict
            The indices of the A_matrix, keyed by the isotope name
        inv_iso_indices : dict
            The inverse of iso_indices, where the key is the index
        duration : float
            Timestep length, in units of days
        depletion_step: int
            The timestep when this function is executed
        num_substeps : int
            The number of substeps to use within the depletion solver when
            performing the depletion
        is_serial : bool, optional
            If this is a serial computation, then we want to copy the results
            from the temp dir to the working dir. Defalts to True

        Returns
        -------
        new_isos : Iterable of 3-tuple (str, str, bool)
            A list of the isotope name (str), the xs library (str), and whether
            it is depleting (bool). There is one of these 3-tuples per isotope.
        new_fracs : np.ndarray
            A 1-D vector containing the depleted atom fractions for each of
            the isotopes in new_isos
        new_density : float
            The new material density in units of a/b-cm
        """

        raise NotImplementedError

    def clear_logs(self):
        self.logs = []
