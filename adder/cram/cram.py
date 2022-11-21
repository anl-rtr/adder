import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from adder.material import Material
from adder.depletion import Depletion
from adder.type_checker import *
from .constants import *


class CRAMDepletion(Depletion):
    """The class providing the interfaces to the internal Chebyshev
    rational approximation method (CRAM) solver as the depletion engine
    of choice.

    Parameters
    ----------
    exec_cmd : str
        The command needed to run the neutronics solver
    num_threads : int
        Number of shared memory threads to utilize
    num_procs : int
        Number of distributed memory instances to utilize
    chunksize : int
        Number of depleting materials to load at a time per thread
    order : {16, 48}
        The CRAM order to use when solving. The larger order, the more
        computational expense but at better accuracy especially at
        longer irradiation times.

    Attributes
    ----------
    solver_type : str
        A shorthand for the solver represented by this instance
    exec_cmd : str
        The command needed to run the neutronics solver, including any
        MPI commands
    num_threads : int
        Number of shared memory threads to utilize
    num_procs : int
        Number of distributed memory instances to utilize
    chunksize : int
        Number of depleting materials to load at a time per thread
    order : {16, 48}
        The CRAM order to use when solving. The larger order, the more
        computational expense but at better accuracy especially at
        longer irradiation times.
    alpha : numpy.ndarray
        Complex residues of poles :attr:`theta` in the incomplete partial
        factorization. Denoted as :math:`\tilde{\alpha}`
    theta : numpy.ndarray
        Complex poles :math:`\theta` of the rational approximation
    alpha0 : float
        Limit of the approximation at infinity
    decay_data : various
        Pre-computed decay library/matrix data, with a specific type depending
        on the solver.

    """

    def __init__(self, exec_cmd, num_threads, num_procs, chunksize, order):
        self.order = order
        if order == 16:
            self.alpha = c16_alpha
            self.theta = c16_theta
            self.alpha0 = c16_alpha0
        elif order == 48:
            self.alpha = c48_alpha
            self.theta = c48_theta
            self.alpha0 = c48_alpha0

        # Note that exec_cmd has no real meaning here since there is
        # no command line argument to make
        super().__init__(exec_cmd, num_threads, num_procs, chunksize)

    @property
    def solver(self):
        return "cram" + str(self.order)

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        check_value("order", order, ORDERS)
        self._order = order

    def compute_decay(self, depl_lib):
        """Computes the decay library. For ORIGEN this is the lib string, for
        CRAM this is the decay matrix"""

        self.decay_data = sp.csr_matrix(depl_lib.build_decay_matrix(),
                                        dtype=np.float64)

    def compute_library(self, depl_lib, flux):
        """Computes and returns the total library,
        decay + flux-induces reactions. For ORIGEN this is a lib string,
        for CRAM this is the sparse matrix"""

        if self.decay_data is None:
            self.compute_decay(depl_lib)
        A_matrix = \
                depl_lib.build_depletion_matrix(flux, "csr", self.decay_data)
        return A_matrix

    def _eval_expm(self, A, n0, dt, dt_units="d"):
        """Evaluate the matrix-exponential (depletion equations) using
        IPF CRAM

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Sparse transmutation matrix ``A[j, i]`` desribing rates at
            which isotope ``i`` transmutes to isotope ``j``
        n0 : numpy.ndarray
            Initial compositions, typically given in number of atoms in
            some material or an atom density
        dt : float
            Time of the specific interval to be solved
        dt_units {"d", or "s"}
            The units that the time is provided in

        Returns
        -------
        numpy.ndarray
            Final compositions after ``dt``

        """

        # This method is adapted from an equivalent in OpenMC
        # (https://github.com/openmc-dev/openmc)
        # OpenMC is distributed with an MIT/X license and is provided with the
        # following disclaimer:
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
        # FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
        # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
        # IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
        # CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

        # Convert dt from days to seconds
        if dt_units == "d":
            dt_sec = dt * 86400.
        else:
            dt_sec = dt
        A = sp.csr_matrix(A * dt_sec, dtype=np.float64)
        y = np.asarray(n0, dtype=np.float64)
        ident = sp.eye(A.shape[0])
        for alpha, theta in zip(self.alpha, self.theta):
            y += 2. * np.real(alpha * sla.spsolve(A - theta * ident, y))
        return y * self.alpha0

    def execute(self, material, A_matrix, iso_indices, inv_iso_indices,
                duration, depletion_step, num_substeps, is_serial=True):
        """Execute the Depletion solver on each of the materials provided.

        Parameters
        ----------
        material : adder.Material
            The problem material
        A_matrix : scipy.csr_matrix
            The depletion matrix from the applicable depletion library
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

        # Create the composition's starting vector
        n_0 = material.get_library_number_density_vector(iso_indices)

        # Now we simply can run our solver
        dt = duration / num_substeps
        for _ in range(num_substeps):
            n_out = self._eval_expm(A_matrix, n_0, dt)
            n_0 = n_out

        new_isos, new_fracs, new_density = \
            material.calc_composition_from_number_densities(
                n_out, iso_indices, inv_iso_indices)

        return new_isos, new_fracs, new_density
