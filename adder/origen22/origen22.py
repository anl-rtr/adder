from collections import OrderedDict
import os
import tempfile
import shutil
import subprocess
import zlib

import numpy as np

import adder.data
from adder.material import Material
from adder.depletion import Depletion
from adder.constants import ORIGEN_ISO_DECAY_TYPES
from .constants import *
from .write_input import write_input
from .read_results import get_tape7
from .depletionlibrary_origen import to_origen, make_origen_decay_lib, \
    make_origen_xs_nfy_lib


class Origen22Depletion(Depletion):
    """The class providing the interfaces to ORIGEN2.2 as the depletion
    engine of choice.

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
    decay_data : various
        Pre-computed decay library/matrix data, with a specific type depending
        on the solver.
    """

    def __init__(self, exec_cmd, num_threads, num_procs, chunksize):
        super().__init__(exec_cmd, num_threads, num_procs, chunksize)
        self.isotope_types = ORIGEN_ISO_DECAY_TYPES

    def compute_decay(self, depl_lib):
        """Computes the decay library. For ORIGEN this is the lib string, for
        CRAM this is the decay matrix"""

        self.decay_data = zlib.compress(
            make_origen_decay_lib(depl_lib, self.isotope_types).encode())

    def compute_library(self, depl_lib, flux):
        """Computes and returns the total library,
        decay + flux-induces reactions. For ORIGEN this is a lib string,
        for CRAM this is the sparse matrix"""

        if self.decay_data is None:
            self.compute_decay(depl_lib)
        rxn_lib = make_origen_xs_nfy_lib(depl_lib, self.isotope_types, flux)
        return "".join((zlib.decompress(self.decay_data).decode(), rxn_lib))

    @property
    def solver(self):
        return "origen2.2"

    @staticmethod
    def assign_isotope_types(depl_lib):
        """ORIGEN2 requires discriminating isotopes by type. This method
        traverses the chain and identifies the types of all isotopes

        Note this is executed before writing an ORIGEN library;
        these types are applied consistently across the decay and xs
        libraries.

        Parameters
        ----------
        depl_lib : DepletionLibrary
            The depletion library to utilize.

        Returns
        -------
        isotope_types : dict
            Dictionary with keys of "actinide", "activation", and "fp" where
            each is the set of isotope names in that category of isotope type.
        """

        # This function is not yet production-ready. To make it so,
        # we would need to perform TTA to limit the number of isotopes
        # In such a case there would be no need for ORIGEN and so this
        # makes no sense. Instead we will leave this here as it does
        # enable small libraries to be built without regard for
        # what isotopes exist for each type of an ORIGEN library.

        # First we identify the starting points for the chains
        # Start with actinides and fps
        actinide_start = set()
        fp_start = set()
        for iso_name, iso_obj in depl_lib.isotopes.items():
            # Consistent with sec 5 of the ORIGEN manual, actinides are
            # those isotopes whose Z > 90
            z, _, _ = adder.data.zam(iso_name)
            if z >= 90:
                actinide_start.add(iso_name)

            # Its still possible for a fission xs to be present (e.g.,
            # all actinium and ra-226 have non-zero fission x/s in
            # ENDF/B-VII.1)
            # We dont have this is an 'else' branch so that we are still
            # computing sigmas[...] for the nfy block
            xs = iso_obj.neutron_xs
            # If its' sigma_(n,3n) > sigma_(n,a) and
            # sigma_f > sigma_(n,p) then it is an actinide
            # Init and then get the cross sections
            sigmas = {"(n,3n)": 0., "(n,a)": 0., "fission": 0.,
                      "(n,p)": 0.}
            all_zero = True
            if xs is not None:
                for xs_type in sigmas:
                    val = xs[xs_type]
                    if val is not None:
                        # Then just peel off the 1-group
                        # (unit flux weighted) xs from the
                        # (xs, targets_, yields_, q_value) tuple
                        sigmas[xs_type] = np.sum(val[0])
                        if sigmas[xs_type] >= 0.:
                            all_zero = False
            # Now do the actinide evaluation
            if not all_zero and (sigmas["(n,3n)"] >=
                                 sigmas["(n,a)"]) and (sigmas["fission"] >=
                                                       sigmas["(n,p)"]):
                actinide_start.add(iso_name)

            # Now find fission products
            # Note this is coded in a way so technically an isotope can
            # be an actinide and a fp. This is non-physical but good for
            # dummy chains used for testing.
            nfy = iso_obj.neutron_fission_yield
            decay = iso_obj.decay
            has_sf_decay = False
            if decay is not None:
                # Filter out stable isotopes
                if decay.decay_constant > 0.:
                    # Filter out onnly those with sf decays
                    if "sf" in decay:
                        sf_data = decay["sf"]
                        # Make sure the branching ratio isn't 0
                        # the BR is the 0th index of the sf_data tuple
                        if sf_data[0] > 0.:
                            has_sf_decay = True

            if nfy is not None and (sigmas['fission'] > 0. or has_sf_decay):
                # Then we simply add all the fission products with yield > 0
                # to the fp_start set
                for fp_name, fp_yield in nfy.items():
                    if fp_yield > 0.:
                        fp_start.add(fp_name)

        # Now I have classified the isotopes amongst actinides and fps.
        # Next I need to build the chains for each of these.

        # Define the function to be used to build the chain
        def _find_chain(start, depl_lib, is_actinide):
            # Start is the starting isotope

            def _get_next_level(parent, depl_lib, is_actinide):
                next_level = set()
                nxs = depl_lib.isotopes[parent].neutron_xs
                dk = depl_lib.isotopes[parent].decay

                # Now find all the children, first from neutron rxns
                if nxs is not None:
                    # Then get all targets with non-zero xs from all valid
                    # reaction channels. Note that a valid reaction channel
                    # is dependent on if this is an actinide or not.
                    valid_types = ("(n,gamma)", "(n,2n)")
                    if is_actinide:
                        # Here we ignore fission since those children
                        # have been treated as the start of the fp set
                        valid_types = valid_types + ("(n,3n)",)
                    else:
                        valid_types = valid_types + ("(n,a)", "(n,p)")
                    for xs_type in nxs.keys():
                        if xs_type in valid_types:
                            xs, targets, yields, _ = nxs[xs_type]
                            onegrp_xs = np.sum(xs)
                            if onegrp_xs > 0.:
                                for t in range(len(targets)):
                                    if yields[t] > 0.:
                                        next_level.add(targets[t])

                # Now do the same for the decays
                if dk is not None:
                    # ignoring spontaneous fission since used as the start
                    # of the fp set
                    valid_types = ("beta-", "ec/beta+", "alpha", "it",
                                   "beta-,n")
                    if dk.decay_constant > 0.:
                        # Now just look for all non-zero yield targets
                        for dk_type in dk.keys():
                            # Filter to only the valid rxn types
                            if dk_type in valid_types:
                                br, targets, yields = dk[dk_type]
                                if br > 0.:
                                    for t in range(len(targets)):
                                        if yields[t] > 0.:
                                            next_level.add(targets[t])
                return next_level

            # Initialize the chain members set
            chain_members = set([start])

            # Keep track of which isotopes we have visited so I know
            # when to exit on a circular loop
            isotopes_visited = {
                k: False for k in depl_lib.isotope_indices.keys()}

            # Set the starting point
            generation = set([start])
            # And now traverse the tree finding all the children in the
            # generations below start
            while len(generation) > 0:
                # Get the next generation from each of the current gen
                next_gen = set()
                for item in generation:
                    # Only do it if we have not yet found the children
                    # of this isotope (avoiding infinite loops)
                    if not isotopes_visited[item]:
                        next_gen.update(_get_next_level(item, depl_lib,
                                                        is_actinide))
                        isotopes_visited[item] = True
                # next gen now includes all the children from the starting
                # generation
                # Update our chain with this next generation
                chain_members.update(next_gen)

                # And set the next generation
                generation = next_gen

            return chain_members

        # Now we can get all the actinides
        actinide = set()
        for start_name in actinide_start:
            actinide.update(_find_chain(start_name, depl_lib, True))
        # Ditto for fps
        fp = set()
        for start_name in fp_start:
            fp.update(_find_chain(start_name, depl_lib, False))

        # Finally, our activation sarting point is everything *not* in
        # actinide or fp
        activation_start = (set(depl_lib.isotopes.keys()) - actinide) - fp
        # Now we repeat building this chain
        activation = set()
        for start_name in activation_start:
            activation.update(_find_chain(start_name, depl_lib, False))

        isotope_types = \
            {"activation": activation, "actinide": actinide, "fp": fp}
        return isotope_types

    def execute(self, material, lib_str, iso_indices, inv_iso_indices,
                duration, depletion_step, num_substeps, is_serial=True):
        """Execute the Depletion solver on each of the materials provided.

        Parameters
        ----------
        material : adder.Material
            The problem material
        lib_str : str
            The pre-genereated ORIGEN library to write to disk
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

        # Do the work in a temporary directory
        start_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                N_out = _worker(self.exec_cmd, self.isotope_types, material,
                    lib_str, iso_indices, inv_iso_indices, duration,
                    num_substeps, temp_dir)
                new_isos, new_fracs, new_density = \
                    material.calc_composition_from_number_densities(
                        N_out, iso_indices, inv_iso_indices)
            except Exception as e:
                if is_serial:
                    # Get the ORIGEN ins/outs and copy somewhere permanent
                    dest = os.path.join(start_dir, material.name)
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(temp_dir, dest)

                    # And propagate the error
                raise e
        return new_isos, new_fracs, new_density


def _worker(exec_name, isotope_types, material, lib_str, iso_indices,
            inv_iso_indices, duration, num_substeps, temp_path):
    # Get the number density vctor and convert to mols
    n_vector = \
        material.get_library_number_density_vector(
            iso_indices) / adder.constants.AVOGADRO

    mat_name = material.name

    # And provide a dictionary with the key is the origen nucid and the
    # value is the moles; this is to match the format of the write_input
    # method.
    composition = OrderedDict()
    for i in range(len(n_vector)):
        if n_vector[i] > 0.:
            # Then we will want it, get the origen nucid key
            iso_name = inv_iso_indices[i]
            z, a, m = adder.data.zam(iso_name)
            origen_nucid = "{}".format(10000 * z + 10 * a + m)

            # And set the value
            composition[origen_nucid] = n_vector[i]

    # Write the library
    lib_path = os.path.join(temp_path, "TAPE{}.INP".format(XS_LIB_UNIT))
    with open(lib_path, mode="w") as file:
        file.write(lib_str)

    # Write the non-standard reaction info
    # TODO: When ORIGEN non-std rxns are understood, re-enable this
    # tape9_path = temp_path / "TAPE{}.INP".format(SUBXS_LIB_UNIT)
    # num_nonstd_rxn = \
    #     create_nonstandard_rxn_lib(depl_lib, str(tape9_path))
    num_nonstd_rxn = 0

    # Create the input file
    tape5 = write_input(mat_name, composition, duration, "days",
                        num_substeps, material.flux_1g, isotope_types,
                        num_nonstd_rxn, verbose_output=False)

    # Write the input files
    tape5_path = os.path.join(temp_path, "TAPE5.INP")
    with open(tape5_path, mode="w") as input_file:
        input_file.write(tape5)

    # Execute ORIGEN
    subprocess.run([exec_name], stdout=subprocess.PIPE, check=True,
                   stderr=subprocess.STDOUT, text=True, shell=False,
                   cwd=temp_path, timeout=30)

    # Get the results
    tape7_path = os.path.join(temp_path, "TAPE7.OUT")
    results = get_tape7(tape7_path)

    # Now convert the results dict (just like composition), to a number
    # density vector
    n_out = np.zeros(len(iso_indices))
    for origen_nucid, mol in results.items():
        z = int(origen_nucid) // 10000
        am = int(origen_nucid) % 10000
        a = am // 10
        m = am % 10

        iso_name = adder.data.gnd_name(z, a, m)
        vector_idx = iso_indices[iso_name]
        n_out[vector_idx] = mol * adder.constants.AVOGADRO

    return n_out
