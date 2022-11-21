from collections import OrderedDict
from collections.abc import Iterable
import subprocess
import shlex

import numpy as np

from adder.isotope import update_isotope_depleting_status
from adder.material import Material
from adder.loggedclass import LoggedClass
from adder.type_checker import *
from adder.constants import IN_CORE


class Neutronics(LoggedClass):
    """The class which contains the data and methods needed for
    interfacing with the neutronics solver.

    This class should be extended for each and every neutronics solver
    for which support is included in Adder.

    Parameters
    ----------
    mpi_cmd : str
        The command used to run MPI (blank str if none)
    exec_cmd : str
        The command needed to run the neutronics solver
    base_input_filename : str
        Path and filename of the initial neutronics solver input file
    num_threads : int
        Number of shared memory threads to utilize
    num_procs : int
        Number of distributed memory instances to utilize
    use_depletion_library_xs : bool
        Whether or not to use the depletion library's cross sections
    reactivity_threshold : float
        The threshold to apply when determining whether or not to
        incorporate an isotope in the neutronics model
    reactivity_threshold_initial : bool
        Whether or not to apply the reactivity_threshold to the initial
        inventory

    Attributes
    ----------
    mpi_cmd : str
        The command used to run MPI (blank str if none)
    solver : str
        A shorthand for the solver interfaced with by this instance
    exec_cmd : str
        The command needed to run the neutronics solver, including any
        MPI commands
    base_input_filename : str
        Path and filename of the initial neutronics solver input file
    num_threads : int
        Number of shared memory threads to utilize
    num_procs : int
        Number of distributed memory instances to utilize
    use_depletion_library_xs : bool
        Whether or not to use the depletion library's cross sections
    base_input : None or OrderedDict
        The contents of the base input neutronics solver file.
    inputs : Iterable of OrderedDict
        The input files produced at each time; the dimension of the
        iterable is the time index, and the dictionary contains the
        contentsof the neutronics input file.
    neutronics_isotopes : OrderedDict
        Allowed isotopes with neutronics solver identifiers
        (e.g., 92235.70c) as keys, values are the atomic-weight-ratios.
    reactivity_threshold : float
        The threshold to apply when determining whether or not to
        incorporate an isotope in the neutronics model
    reactivity_threshold_initial : bool
        Whether or not to apply the reactivity_threshold to the initial
        inventory
    """

    VALID_SHUFFLE_TYPES = ("material")
    VALID_TRANSFORM_TYPES = tuple()

    def __init__(self, mpi_cmd, exec_cmd, base_input_filename, num_threads,
                 num_procs, use_depletion_library_xs, reactivity_threshold,
                 reactivity_threshold_initial):
        self.exec_cmd = exec_cmd
        self.base_input_filename = base_input_filename
        self.num_threads = num_threads
        self.num_procs = num_procs
        # mpi_cmd should be initialized after num_procs since the
        # mpi_cmd setter uses num_procs for a warning
        self.mpi_cmd = mpi_cmd
        self.use_depletion_library_xs = use_depletion_library_xs
        self.constant_use_depletion_library_xs = use_depletion_library_xs
        self.base_input = None
        self.inputs = []
        self.neutronics_isotopes = OrderedDict()
        self.reactivity_threshold = reactivity_threshold
        self.reactivity_threshold_initial = reactivity_threshold_initial

        # Set up the logger and log that we initialized our Neutronics
        # solver
        super().__init__(6, __name__)
        msg = "Initialized Neutronics Solver, {}".format(self.solver)
        self.log("info", msg, 2)
        msg = "Neutronics Command: {}".format(self.exec_cmd)
        self.log("info", msg, 4)

        # Test the neutronics commands
        if self.solver != "test":
            # First check the MPI command
            if len(self.mpi_cmd) > 0 and self.num_procs >= 1:
                try:
                    cmd = shlex.split("which {}".format(self.mpi_cmd))
                    status = subprocess.run(cmd, check=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        universal_newlines=True, shell=False)
                    msg = "Neutronics MPI Location: {}".format(
                        status.stdout.strip())
                    self.log("info_file", msg)
                except subprocess.CalledProcessError:
                    self.log("error", "Could Not Find Provided MPI!")
            # Now check the neutronic solver itself
            try:
                cmd = shlex.split("which {}".format(self.exec_cmd))
                status = subprocess.run(cmd, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True, shell=False)
                msg = "Neutronics Solver Location: {}".format(
                    status.stdout.strip())
                self.log("info_file", msg)
            except subprocess.CalledProcessError:
                self.log("error", "Could Not Find Neutronics Solver!")

    @property
    def solver(self):
        return "test"

    @property
    def mpi_cmd(self):
        return self._mpi_cmd

    @mpi_cmd.setter
    def mpi_cmd(self, mpi_cmd):
        check_type("mpi_cmd", mpi_cmd, str)
        if len(mpi_cmd) == 0 and self.num_procs > 1:
            msg = "An MPI command is not provided, however the number of" + \
                " MPI processes is > 1 ({})".format(self.num_procs)
            raise ValueError(msg)
        self._mpi_cmd = mpi_cmd

    @property
    def exec_cmd(self):
        return self._exec_cmd

    @exec_cmd.setter
    def exec_cmd(self, exec_cmd):
        check_type("exec_cmd", exec_cmd, str)
        self._exec_cmd = exec_cmd

    @property
    def base_input_filename(self):
        return self._base_input_filename

    @base_input_filename.setter
    def base_input_filename(self, base_input_filename):
        check_type("base_input_filename", base_input_filename, str)
        self._base_input_filename = base_input_filename

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
    def reactivity_threshold(self):
        return self._reactivity_threshold

    @reactivity_threshold.setter
    def reactivity_threshold(self, reactivity_threshold):
        check_type("reactivity_threshold", reactivity_threshold, float)
        check_greater_than("reactivity_threshold", reactivity_threshold, 0.,
                           equality=True)
        self._reactivity_threshold = reactivity_threshold

    @property
    def reactivity_threshold_initial(self):
        return self._reactivity_threshold_initial

    @reactivity_threshold_initial.setter
    def reactivity_threshold_initial(self, reactivity_threshold_initial):
        check_type("reactivity_threshold_initial",
                   reactivity_threshold_initial, bool)
        self._reactivity_threshold_initial = reactivity_threshold_initial

    @property
    def use_depletion_library_xs(self):
        return self._use_depletion_library_xs

    @use_depletion_library_xs.setter
    def use_depletion_library_xs(self, use_depletion_library_xs):
        check_type("use_depletion_library_xs", use_depletion_library_xs, bool)
        self._use_depletion_library_xs = use_depletion_library_xs

    @property
    def base_input(self):
        return self._base_input

    @base_input.setter
    def base_input(self, base_input):
        if base_input is not None:
            check_type("base_input", base_input, OrderedDict)
        self._base_input = base_input

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        # Since check_iterable_type allows a string as an iterable,
        # check to ensure it is *not* a string first
        if isinstance(inputs, str):
            msg = 'Unable to set "inputs" which is not of type "str"'
            self.log("error", msg)
        # Ok, it isnt a string, carry on
        check_iterable_type("inputs", inputs, str)
        self._inputs = inputs

    def init_model(self, materials, additional_info):
        """This method allows downstream classes to do what they need to
        build the model, assigns names, etc. This then initializes the
        statuses of these objects.

        Parameters
        ----------
        materials : Iterable of Material
            A constructed Material object for each material in the
            model; since no is_depleting (or isotope is_depleting)
            information is available at this stage, this must be
            overwritten upstream.
        additional_info : OrderedDict
            The key is the id of interest and the value is the
            user-specified name (used downstream for universes)

        """

        raise NotImplementedError

    def init_h5_file(self, h5_file):
        """Initializes the HDF5 file so subsequent steps can be written
        as they are completed.
        """

        # Store the runtime options as root attributes
        h5_file.attrs["neutronics_solver"] = np.string_(self.solver)
        h5_file.attrs["neutronics_mpi_cmd"] = np.string_(self.mpi_cmd)
        h5_file.attrs["neutronics_exec"] = np.string_(self.exec_cmd)
        h5_file.attrs["base_neutronics_input_filename"] = \
            np.string_(self.base_input_filename)
        h5_file.attrs["num_neutronics_threads"] = self.num_threads
        h5_file.attrs["num_mpi_procs"] = self.num_procs
        h5_file.attrs["use_depletion_library_xs"] = \
            np.bool_(self.use_depletion_library_xs)
        h5_file.attrs["reactivity_threshold"] = self.reactivity_threshold
        h5_file.attrs["reactivity_threshold_initial"] = \
            np.bool_(self.reactivity_threshold_initial)

    def read_input(self, library_data, num_neutron_groups, user_mats_info,
                   user_univ_info, shuffled_mats, shuffled_univs, depl_libs):
        """Return parsed information about the neutronics input file,
        including all the information needed to initialize
        Material objects.

        This method parses the given file to identify separate input
        blocks, and gather pertinent cell and material information. All
        comments will be removed and line continuations also removed.

        Parameters
        ----------
        library_data : str
            The filename and path to the library data
        num_neutron_groups : int
            The number of energy groups
        user_mats_info : OrderedDict
            The keys are the ids in the neutronics solver and
            the value is an OrderedDict of the name, depleting boolean,
            ex_core_status, non_depleting_isotopes list, and
            use_default_depletion_library flag.
        user_univ_info : OrderedDict
            The keys are the universe ids in the neutronics solver and
            the value is an OrderedDict of the name.
        shuffled_mats : set
            The set of material names that are shuffled
        shuffled_univs : set
            The set of universe names that are shuffled
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model

        Returns
        -------
        materials : Iterable of Material
            A constructed Material object for each material in the
            model; since no is_depleting (or isotope is_depleting)
            information is available at this stage, this must be
            overwritten upstream.

        """

        # Set default testing values
        input_file = OrderedDict()
        input_file["options"] = ["option 1", "option 2"]
        input_file["runmode"] = ["fast"]
        self.base_input = input_file

        # Create two test material objects
        name = "1"
        id_ = 1
        density = 1.
        # Differentiate for our two unit test types
        if library_data == "test_lib_file.txt":
            # Then this is the test_reactor case
            isotope_data = [("H1", "70c", False), ["U235", "70c"],
                            ["U238", "72c"]]
            atom_fractions = [4., 5., 1.]
        else:
            # Then this is an integral test
            isotope_data = [("U235", "70c")]
            atom_fractions = [1.]
        is_depleting = True
        default_xs_library = "71c"
        thermal_xs_libraries = []
        mat1 = Material(name, id_, density, isotope_data, atom_fractions,
                        is_depleting, default_xs_library, 1,
                        thermal_xs_libraries, IN_CORE)

        if id_ in user_mats_info:
            mat1.name = user_mats_info[id_]["name"]
            mat1.is_depleting = user_mats_info[id_]["depleting"]
            if "density" in user_mats_info[id_] and user_mats_info[id_]["density"] is not None:
                mat1.density = user_mats_info[id_]["density"]
            if "volume" in user_mats_info[id_] and user_mats_info[id_]["volume"] is not None:
                mat1.volume = user_mats_info[id_]["volume"]
            if not self.use_depletion_library_xs:
                mat1.is_default_depletion_library = \
                    user_mats_info[id_]["use_default_depletion_library"]
            else:
                mat1.is_default_depletion_library = True
            key = "non_depleting_isotopes"
            if len(user_mats_info[id_][key]) > 0:
                # Then we have specific values
                for i, iso in enumerate(mat1.isotopes):
                    if iso.name in user_mats_info[id_][key]:
                        mat1.isotopes[i] = \
                            update_isotope_depleting_status(iso, False)
            key = "reactivity_threshold_initial"
            if key in user_mats_info[id_]:
                mat_reactivity_threshold_initial = \
                    user_mats_info[id_][key]
            else:
                mat_reactivity_threshold_initial = \
                    self.reactivity_threshold_initial
            mat1.establish_initial_isotopes(mat_reactivity_threshold_initial)

        name = "2"
        id_ = 2
        if library_data != "test_lib_file.txt":
            atom_fractions = [2.]
        mat2 = Material(name, id_, density, isotope_data,
                        atom_fractions, is_depleting,
                        default_xs_library, 1, thermal_xs_libraries, IN_CORE)
        if id_ in user_mats_info:
            mat2.name = user_mats_info[id_]["name"]
            mat2.is_depleting = user_mats_info[id_]["depleting"]
            if "density" in user_mats_info[id_] and user_mats_info[id_]["density"] is not None:
                mat2.density = user_mats_info[id_]["density"]
            if "volume" in user_mats_info[id_] and user_mats_info[id_]["volume"] is not None:
                mat2.volume = user_mats_info[id_]["volume"]
            if not self.use_depletion_library_xs:
                mat2.is_default_depletion_library = \
                    user_mats_info[id_]["use_default_depletion_library"]
            else:
                mat2.is_default_depletion_library = True
            key = "non_depleting_isotopes"
            if len(user_mats_info[id_][key]) > 0:
                # Then we have specific values
                for i, iso in enumerate(mat2.isotopes):
                    if iso.name in user_mats_info[id_][key]:
                        mat2.isotopes[i] = \
                            update_isotope_depleting_status(iso, False)
            key = "reactivity_threshold_initial"
            if key in user_mats_info[id_]:
                mat_reactivity_threshold_initial = \
                    user_mats_info[id_][key]
            else:
                mat_reactivity_threshold_initial = \
                    self.reactivity_threshold_initial
            mat2.establish_initial_isotopes(mat_reactivity_threshold_initial)

        materials = [mat1, mat2]

        for mat in materials:
            self.update_logs(mat.logs)
            mat.clear_logs()

        return materials

    def execute(self, filename, label, materials, depl_libs, store_input,
                deplete_tallies, user_tallies, user_output, fast_forward,
                update_iso_status=True):
        """Writes the input file, executes the neutronics solver, and
        then reads the results

        Parameters
        ----------
        filename : str
            The input file to write to.
        label : str
            Label to append to the title
        materials: Iterable of Materials
            New material information to include.
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
        store_input : bool
            Whether or not to store the input file in :attrib:`inputs`
        deplete_tallies : bool
            Whether or not to write the tallies needed for depletion
        user_tallies : bool
            Whether or not to write the user tallies
        user_output : bool
            Whether or not to write the user's output control cards
        fast_forward : bool
            Whether or not to use existing output files of the correct names
            (True) or re-calculate.
        update_iso_status : bool, optional
            Whether or not to update the status of the isotopes_in_neutronics
            parameter of each material; this is not needed for iterative
            analyses such as critical searches and therefore need not be
            performed again. Defaults to updating the status
        """

        inp_name = filename + ".inp"
        self.write_input(inp_name, label, materials, depl_libs, store_input,
                         deplete_tallies, user_tallies, user_output,
                         update_iso_status)

        self._exec_solver(inp_name, filename, fast_forward=fast_forward)

        keff, keff_uncertainty, flux, nu, volumes = \
            self._read_results(filename, materials, depl_libs, deplete_tallies)

        return keff, keff_uncertainty, flux, nu, volumes

    def write_input(self, filename, label, materials, depl_libs, store_input,
                    deplete_tallies, user_tallies, user_output,
                    update_iso_status=True):
        """Updates the input for the particular run case.

        Parameters
        ----------
        filename : str
            The file to write to, without the suffix
        label : str
            Label to append to the title
        materials: Iterable of Materials
            New material information to include.
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
        store_input : bool
            Whether or not to store the input file in :attrib:`inputs`
        deplete_tallies : bool
            Whether or not to write the tallies needed for depletion
        user_tallies : bool
            Whether or not to write the user tallies
        user_output : bool
            Whether or not to write the user's output control cards
        update_iso_status : bool, optional
            Whether or not to update the status of the isotopes_in_neutronics
            parameter of each material; this is not needed for iterative
            analyses such as critical searches and therefore need not be
            performed again. Defaults to updating the status
        """

        new_input = "this is a new input file"

        new_input = OrderedDict()
        new_input["options"] = ["option 3", "option 4"]
        new_input["runmode"] = ["slow"]

        if store_input:
            self.inputs.append(new_input)

    def _exec_solver(self, inp_name, out_name, fast_forward=False):
        """Performs the computation."""
        output = "completed"

        with open(out_name, "w+") as file:
            file.write(output)

    def _read_results(self, filename, materials, depl_libs, deplete_tallies):
        """Gets the flux and volume of each material after the current
        neutronics solve. The material object is *not* updated; that is
        saved for the main adder code to handle since it is solver agnostic.

        Parameters
        ----------
        filename : str
            The filename that was used when executing the solver
        materials : Iterable of Material
            The problem materials, providing the IDs of their constituent
            cells
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
        deplete_tallies : bool
            Whether or not depletion tallies are included in the model

        Returns
        -------
        keff : float
            Calculated k-eigenvalue of the problem.
        keff_uncertainty : float
            Estimated k-eigenvalue uncertainty
        flux : OrderedDict of np.ndarray
            Dictionary where the key is the material id and the value is the
            group-wise tallied flux mean.
        nu : float
            average number of neutrons produced per fission.
        volumes : OrderedDict
            The material id is the key and the value is a list of
            volumes in cm^3

        """

        keff = 2.0
        keff_uncertainty = 0.01
        flux = OrderedDict()
        flux[1] = np.array([3. / 2.])
        flux[2] = np.array([2. / 3.])
        nu = 2.4
        volumes = OrderedDict()
        volumes[1] = 0.4
        volumes[2] = 0.6

        return keff, keff_uncertainty, flux, nu, volumes

    def calc_volumes(self, materials, vol_data, target_unc, max_hist,
                     only_depleting):
        """Computes the volumes of the materials of interest, storing
        them on the Material objects

        Parameters
        ----------
        materials : Iterable of Material
            The problem materials, providing the IDs of their
            constituent cells
        vol_data : dict
            Parameters for the sampling volume
        target_unc : float
            The target uncertainty, in percent, for which the stochastic
            volume estimation will meet on every region's volume.
        max_hist : int
            The maximum number of histories to run in the stochastic
            volume estimation.
        only_depleting : bool
            Whether or not to iterate on volumes for only depleting
            materials; note this can significantly increase the runtime
            if set to False.

        Returns
        -------
        volume : OrderedDict
            The material id is the key and the value is volume in cm^3
        """

        raise NotImplementedError("Only Implemented in Extended Types!")

    def shuffle(self, start_name, to_names, shuffle_type, materials, depl_libs):
        """Shuffles the object named start_name to the locations listed
        in to_names. This will include handling the cases where the
        start_name object is in storage, supply, or in-core.

        Parameters
        ----------
        start_name : str
            The name of the item (type shuffle_type) to move
        to_names : Iterable of str
            The names of the objects (of type shuffle_type) to be
            displaced. The 0th entry is the location for start_name to
            be moved to. The 1st entry is for the item displaced by that
            first move, so on and so forth. The last item displaced will
            head to storage.
        shuffle_type : str
            The type of shuffle to perform; this depends on the actual
            neutronics solver type; this could be "material" or
            "universe", for example.
        materials : List of Material
            The materials to work with
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model

        """

        # Check the types
        check_type("start_name", start_name, str)
        check_type("to_names", to_names, list)
        check_iterable_type("to_names", to_names, str)
        check_value("shuffle_type", shuffle_type, self.VALID_SHUFFLE_TYPES)
        check_iterable_type("materials", materials, Material)

    def transform(self, names, yaw, pitch, roll, angle_units, matrix,
                  displacement, transform_type, transform_in_place=False):
        """Transforms the universe named `name` according to the angles in
        yaw, pitch, and roll and the translation in displacement.

        Parameters
        ----------
        names : List of str
            The names of the item to transform
        yaw : None or float
            The angle to rotate about the z-axis. If None, then the matrix is
            used instead.
        pitch : None or float
            The angle to rotate about the y-axis. If None, then the matrix is
            used instead.
        roll : None or float
            The angle to rotate about the x-axis. If None, then the matrix is
            used instead.
        angle_units : {"degrees", "radians"}
            The units of the above angles; ignored if a matrix is provided
        matrix : None or np.ndarray
            The rotation matrix, provided in a 3x3 matrix. If None, then yaw,
            pitch, and roll are used instead.
        displacement : Iterable of float
            The displacement vector to translate the object
        transform_type : str
            The type of object to transform; this depends on the actual
            neutronics solver type; this could be "material" or
            "universe", for example.
        transform_in_place : bool, optional
            If the existing transform should be updated (True) or a new
            transform created. Defaults to False.
        """

        # Perform type checking
        check_type("names", names, Iterable)
        for name in names:
            check_type("name", name, str)
        check_type("yaw", yaw, (type(None), float))
        check_type("pitch", pitch, (type(None), float))
        check_type("roll", roll, (type(None), float))
        check_value("angle_units", angle_units, ("degrees", "radians"))
        check_type("matrix", matrix, (type(None), np.ndarray))
        check_iterable_type("displacement", displacement, float)
        check_length("displacement", displacement, 3)
        check_type("transform_in_place", transform_in_place, bool)

    def geom_search(self, transform_type, axis, names, angle_units, k_target,
                    bracket_interval, target_interval, uncertainty_fraction,
                    initial_guess, min_active_batches, max_iterations,
                    materials, case_idx, operation_idx, depl_libs):
        """Performs a search of geometric transformations to identify that
        which yields the target k-eigenvalue within the specified interval.

        Parameters
        ----------
        transform_type : str
            The type of object to transform; this depends on the actual
            neutronics solver type; this could be "material" or
            "universe", for example.
        axis : constants.VALID_GEOM_SWEEP_AXES
            The axis of interest to sweep
        names : List of str
            The names of the components of type `type_` to apply the
            perturbation to
        angle_units : {"degrees", "radians"}
            The units of the above angles
        k_target : float
            The target k-eigenvalue to search for
        bracket_interval : Iterable of float
            The lower and upper end of the ranges to search.
        target_interval : float
            The range around the k_target for which is considered success
        uncertainty_fraction : float
            This parameter sets the stochastic uncertainty to target when the
            initial computation of an iteration reveals that a viable solution
            may exist.
        initial_guess : float
            The starting point to search
        min_active_batches : int
            The starting number of batches to run to ensure enough keff samples
            so that it follows the law of large numbers. This is the number of
            batches run when determining if a case has a chance of being a
            suitable solution. This only applies to Monte Carlo solvers.
        max_iterations : int
            The total number of search iterations to perform before terminating
            the search.
        materials : List of Material
            The model materials
        case_idx : int
            The case index, for printing
        operation_idx : int
            The operation index, for printing
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model

        Returns
        -------
        converged : bool
            A flag denoting whether or not the solution was converged
        iterations : int
            The number of iterations required
        val_new : float
            The latest control group displacement
        keff_new : float
            The obtained value of k-effective
        keff_std_new : float
            The obtained k-effective (1-sigma) uncertainty

        """

        raise NotImplementedError("Only Implemented in Extended Types!")
