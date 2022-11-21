import numpy as np
from adder.loggedclass import LoggedClass
from adder.constants import BASE_LIB
from adder.material import Material
from adder.isotope import *
from adder.cram import CRAMDepletion
from adder.cram.constants import c16_alpha, c16_alpha0, c16_theta, \
    c48_alpha, c48_alpha0, c48_theta
from adder.type_checker import *
from .msr_system import MSRSystem


SMOOTHING_METHODS = ["histogram", "average"]


class MSRDepletion(CRAMDepletion, LoggedClass):
    """The class which handles depletion calculations of a flowing fuel
    MSR reactor.

    This class will internally execute the CRAM solver as needed.

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
    cram_order : {16, 48}
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
    fluid_mats : List of Material
        The materials that are considered in the flowing fuel flowpath
    systems : List of MSRSystem
        The list of fluid systems present
    flux_smoothing_method : {"histogram", "average"}
        The flux smoothing methodology to apply when the fuel fluid
        flows from one region to another in the core.

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
        self.exec_cmd = exec_cmd
        self.num_threads = num_threads
        self.num_procs = num_procs
        self.chunksize = chunksize
        self.logs = []
        # Set up the logger and log that we initialized our Depletion
        # solver
        LoggedClass.__init__(self, 6, __name__)
        msg = "Initialized Depletion Solver, {}".format(self.solver)
        self.log("info", msg, 2)

    def set_msr_params(self, flux_smoothing_method, solve_method, system_data,
                       materials, depl_libs):
        """Initialize the MSR reactor data.

        Parameters
        ----------
        flux_smoothing_method : {"histogram", "average"}
            The flux smoothing methodology to apply when the fuel fluid
            flows from one region to another in the core.
        solve_method : {"brute", "tmatrix", "tmatrix_expm, "rxn_rate_avg""}
            The methodology to use when performing the depletion
        system_data : List of Dict
            The data for each of the systems modelled
        materials : List of Material
            The materials from the reactor definition
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
        """

        self.systems = []
        self.flux_smoothing_method = flux_smoothing_method
        for system_info in system_data:
            self.systems.append(MSRSystem(solve_method, system_info,
                                          depl_libs, materials))

        # Now get the list of material names that are used by our systems
        self.fluid_mats = []
        for system in self.systems:
            for comp in system.component_network.keys():
                if comp.in_core:
                    self.fluid_mats.append(comp.mat_name)

    @property
    def solver(self):
        return "MSR_cram" + str(self.order)

    @property
    def flux_smoothing_method(self):
        return self._flux_smoothing_method

    @flux_smoothing_method.setter
    def flux_smoothing_method(self, flux_smoothing_method):
        check_value("flux_smoothing_method", flux_smoothing_method,
                    SMOOTHING_METHODS)
        self._flux_smoothing_method = flux_smoothing_method

    def execute(self, materials, depl_libs, duration, depletion_step,
                num_substeps, reactivity_threshold):
        """Execute the Depletion solver on each of the materials
        provided.

        Parameters
        ----------
        materials : Iterable of adder.Material
            The problem materials
        depl_libs : dict
            The set of depletion libraries in use in the problem
        duration : float
            Timestep length, in units of days
        depletion_step: int
            The timestep when this function is executed
        num_substeps : int
            The number of substeps to use within the depletion solver when
            performing the depletion.
        reactivity_threshold : float
            The threshold to apply when determining whether or not to
            incorporate an isotope in the neutronics model

        """

        materials_by_name = {}
        for mat in materials:
            materials_by_name[mat.name] = mat
        num_groups = depl_libs[BASE_LIB].num_neutron_groups

        # Perform the MSR system depletion first
        for system in self.systems:
            self.log("info", "Depleting System: {}".format(system.name),
                     indent=8)

            if self.flux_smoothing_method == "average":
                vol_time_sum = 0.
                avg_flux = np.zeros(num_groups)
                for comp in system.component_network.keys():
                    if comp.in_core:
                        mat = materials_by_name[comp.mat_name]
                        dt = comp.delta_t
                        if dt > duration:
                            V_dt = duration * comp.volume
                        else:
                            V_dt = dt * comp.volume
                        vol_time_sum += V_dt
                        avg_flux += mat.flux * V_dt
                avg_flux /= vol_time_sum
                # Now go back in and assign this avg_flux
                for comp in system.component_network.keys():
                    if comp.in_core:
                        mat = materials_by_name[comp.mat_name]
                        mat.flux = avg_flux

            elif self.flux_smoothing_method == "histogram":
                # The flux is as we need it, so dont do anything
                pass

            # Now set up our starting material
            start_mat = materials_by_name[system.starting_material_name]

            # Create the composition's starting vector from the material
            depl_lib = depl_libs[BASE_LIB]
            n_start = start_mat.get_library_number_density_vector(
                depl_lib.isotope_indices)

            msg = "Performing fluid_depletion with the '{}' method"
            self.log("info_file", msg.format(system.method), indent=8)
            step_duration = duration / num_substeps
            n_step_in = n_start
            for ns in range(num_substeps):
                # Update the flow times for this system
                system.update_time_properties()
                if system.method == "rxn_rate_avg":
                    # In the flux averaging method, a single depletion step
                    # is performed using a time-averaged flux for a given matl
                    # that it would experience as it traverses the loop during
                    # the depletion time step
                    n_step_out = \
                        system.solve_rxn_rate_avg_method(
                            materials_by_name, step_duration, depletion_step,
                            self._eval_expm, n_step_in,
                            depl_lib.atomic_mass_vector)
                else:
                    # Update the matrices with the new fluxes and xs' as well
                    # as the time step length (for t-matrix method)
                    system.compute_matrices(materials_by_name, step_duration,
                                            self._eval_expm)
                    self.log("info_file", "Depletion matrices computed",
                             indent=8)
                    # Perform the solution
                    num_loops = int(
                        np.around(step_duration * 86400. /
                                  system.min_transport_time))
                    msg = "Performing fluid depletion over {} loop transports"
                    self.log("info_file", msg.format(num_loops), indent=8)
                    n_step_out = system.solve(step_duration, depletion_step,
                                              n_step_in, self._eval_expm,
                                              depl_lib.atomic_mass_vector)

                n_step_in = n_step_out
                msg = "Completed Substep {}".format(ns + 1)
                self.log("info_file", msg, indent=8)

            # Now we can update the material data
            n_end = n_step_out
            # Now update the materials, start with the starting mat so
            # we can get our density ratio
            orig_density = start_mat.density
            start_mat.update_from_number_densities(n_end, depl_lib)
            density_ratio = start_mat.density / orig_density
            for comp in system.component_network.keys():
                if comp.in_core:
                    mat = materials_by_name[comp.mat_name]
                    if comp != system.starting_component:
                        mat.update_from_number_densities(n_end, depl_lib,
                                                         density_ratio)
