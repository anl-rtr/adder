import filecmp
import glob
import os
import shutil
import subprocess

import numpy as np
from adder.depletionlibrary import DepletionLibrary, ReactionData, DecayData, \
    YieldData
from tests import default_config as config
from tests import xsdir, xs_files, xs_file_names, xs_file_ext
if config["plot"]:
    import matplotlib.pyplot as plt


ADDER_EXEC = config["exe"]
ORIGEN_EXEC = config["origen_exe"]
MCNP_EXEC = config["mcnp_exe"]
EXTRACT_SCRIPT = config["extract_exe"]


class TestHarness(object):
    """General class for running ADDER tests."""

    def __init__(self, output_ascii_fnames, test_lib_name="test.h5",
                 input_fname="test.add"):
        self._output_ascii_fnames = output_ascii_fnames
        self._test_lib_name = test_lib_name
        self.input_fname = input_fname
        self.results_true_fname = 'results_true.dat'
        self.results_error_fname = 'results_error.dat'
        self.log_messages = None

    def main(self):
        """Accept commandline arguments and either run or update tests."""
        if config['update']:
            self.update_results()
        else:
            self.execute_test()
        self._cleanup()

    def execute_test(self):
        """Run ADDER with the appropriate arguments and check the outputs."""
        try:
            self._create_test_lib()
            self._run_adder()
            results = self._get_results()
            self._write_results(results)
            self._compare_results()
        finally:
            self._cleanup()

    def update_results(self):
        """Update the results_true using the current version of ADDER."""
        try:
            self._create_test_lib()
            self._run_adder()
            results = self._get_results()
            self._write_results(results)
            self._overwrite_results()
        finally:
            self._cleanup()

    def _run_adder(self):
        os.system("{} ./{}".format(ADDER_EXEC, self.input_fname))

    def _get_outputs(self):
        # Reads and combines the MCNP files into a string
        outstr = ""

        for fname in self._output_ascii_fnames:
            with open(fname, "r") as fin:
                outstr += "".join(fin.readlines())

        return outstr

    def _check_log_messages(self):
        """Check for error and warning messages in the ADDER log. The messages
        to be checked are those in self.log_messages. They should tuples of the
        form: ('warning' or 'error', error/warning message, number of expected
        messages)."""

        log_warnings = []
        log_errors = []
        with open("adder.log", "r") as fin:
            for line in fin.readlines():
                if "- WARNING -" in line:
                    log_warnings.append(line)
                elif "- ERROR -" in line:
                    log_errors.append(line)

        # For each message a list is created that contains all the errors or
        # warnings from the ADDER log that contain the message. It is then
        # asserted that the number of times the message appears in the log is
        # the same as what is expected.
        for message in self.log_messages:
            if message[0] == 'warning':
                assert len([s for s in log_warnings if message[1] in s]) == \
                       message[2]
            elif message[0] == 'error':
                assert len([s for s in log_errors if message[1] in s]) == \
                       message[2]

    def _get_results(self):
        """Digest info in the output and return as a string."""
        outstr = self._get_outputs()

        if self.log_messages is not None:
            self._check_log_messages()

        # Process results.h5 if needed
        # ... for inherited classes

        return outstr

    def _write_results(self, results_string):
        """Write the results to an ASCII file."""
        with open('results_test.dat', 'w') as fh:
            fh.write(results_string)

    def _overwrite_results(self):
        """Overwrite the results_true with the results_test."""
        shutil.copyfile('results_test.dat', self.results_true_fname)

    def _compare_results(self):
        """Make sure the current results agree with the _true standard."""
        compare = filecmp.cmp('results_test.dat', self.results_true_fname)
        if not compare:
            os.rename('results_test.dat', self.results_error_fname)
        assert compare, 'Results do not agree.'

    def _cleanup(self):
        """Delete statepoints, tally, and test files."""
        output = glob.glob('*.h5')
        output += glob.glob("*.inp")
        output += glob.glob("case_*")
        output += glob.glob("*.70c")
        output += glob.glob("*.71c")
        output += glob.glob("*.72c")
        output += ['xsdir', 'adder.log', 'results_test.dat']
        for f in output:
            if os.path.exists(f):
                os.remove(f)

    def _create_test_lib(self):
        # This will be a simple depletion library
        depllib = DepletionLibrary("test", np.array([0., 20.]))

        # He4
        he4dk = DecayData(None, "s", 0.)
        depllib.add_isotope("He4", decay=he4dk)

        # U235
        u235xs = ReactionData("b", 1)
        u235xs.add_type("fission", "b", [1.0])
        u235dk = DecayData(None, "s", 200.)
        u235yd = YieldData()
        u235yd.add_isotope("I135", 2. * 0.4)
        u235yd.add_isotope("I134", 2. * 0.6)
        depllib.add_isotope("U235", xs=u235xs, decay=u235dk, nfy=u235yd)

        # I135, stable
        i135dk = DecayData(None, "s", 0.)
        depllib.add_isotope("I135", decay=i135dk)

        # I134, stable
        i134dk = DecayData(None, "s", 0.)
        depllib.add_isotope("I134", decay=i134dk)
        depllib.finalize_library()

        depllib.to_hdf5(self._test_lib_name)

        self._create_ce_data()

    def _create_ce_data(self):
        with open("xsdir", "w") as f:
            f.write(xsdir)

        for ext in xs_file_ext:
            for i, name in enumerate(xs_file_names):
                with open(name.format(ext), "w") as f:
                    f.write(xs_files[i].format(ext))


def get_error(ref_x, ref_y, x, y, scaling=100.):
    interp = np.interp(x, ref_x, ref_y)

    y_err = scaling * (y - interp) / interp

    for i in range(len(y_err)):
        if interp[i] == 0. and y[i] == 0.:
            y_err[i] = 0.

    return y_err


def ref_soln(sig_f, sig_c, nu, p0, q, v, n235_0, n134_0, n135_0, max_t, num_t):
    times = np.linspace(0., max_t, num_t + 1, endpoint=True)
    dt = times[1] - times[0]
    phis = np.zeros_like(times)
    N235 = np.zeros_like(times)
    N134 = np.zeros_like(times)
    N135 = np.zeros_like(times)
    keff = np.zeros_like(times)

    N235[0] = n235_0
    N134[0] = n134_0
    N135[0] = n135_0
    keff[0] = nu * sig_f * N235[0] / (sig_f * N235[0] + sig_c * N134[0])
    phis[0] = p0 / (q * v * sig_f * N235[0])

    for j in range(1, len(times)):
        i = j - 1
        # number density solution uses last flux for computations
        N235[j] = N235[i] * np.exp(-sig_f * phis[i] * dt)
        N134[j] = N134[i] * np.exp(-sig_c * phis[i] * dt) + \
            sig_f / (sig_c - sig_f) * N235[i] * \
            (np.exp(-sig_f * phis[i] * dt) - np.exp(-sig_c * phis[i] * dt))
        N135[j] = n235_0 + n134_0 + n135_0 - N235[j] - N134[j]
        # keff uses current number densities to compute
        keff[j] = nu * sig_f * N235[j] / (sig_f * N235[j] + sig_c * N134[j])
        # phis uses current number densities to compute
        phis[j] = p0 / (q * v * sig_f * N235[j])

    times /= 86400.
    N235 *= 1.E-24
    N134 *= 1.E-24
    N135 *= 1.E-24

    return times, keff, phis, N235, N134, N135


def plot_ref(case_name, times, keff, phis, N235, N134, N135):
    plt.figure(figsize=(10, 8))
    plt.title("Keff vs Time")
    plt.plot(times, keff)
    plt.ylabel("Eigenvalue")
    plt.xlabel("Time [d]")
    plt.savefig(case_name + "_keff.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title("Flux vs Time")
    plt.plot(times, phis)
    plt.ylabel("Flux Level [n/cm^2-s]")
    plt.xlabel("Time [d]")
    plt.savefig(case_name + "_phis.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title("Concentrations vs Time")
    plt.plot(times, N235, label="$^{235}U$")
    plt.plot(times, N134, label="$^{134}I$")
    plt.plot(times, N135, label="$^{135}I$")
    plt.plot(times, N135 + N235 + N134, label="Total")
    plt.ylabel("Concentrations [atom/b-cm]")
    plt.xlabel("Time [d]")
    plt.legend(loc='best')
    plt.savefig(case_name + "_conc.png")
    plt.close()


def get_adder_results():
    # Create the CSV data
    csv_names = ["1"]
    for csv_name in csv_names:
        subprocess.call([EXTRACT_SCRIPT, "results.h5",
                         "mat_{}".format(csv_name), csv_name])

    def get_arr(line, dtype):
        new_line = line.strip("\n, ").split(", ")[1:]
        return [dtype(v) for v in new_line]

    # Now process.
    conc = []
    times = []
    powers = []
    case_labels = []
    op_labels = []
    step_idxs = []
    keffs = []
    keffs_stddev = []
    tot_flux = []
    isos = ["U235", "I134", "I135"]
    for i in range(len(csv_names)):
        with open("mat_{}".format(csv_names[i]), newline='') as csvfile:
            lines = csvfile.readlines()
            if i == 0:
                # Get the metadata
                case_labels = get_arr(lines[1], str)
                op_labels = get_arr(lines[2], str)
                step_idxs = get_arr(lines[3], str)
                times.extend(get_arr(lines[4], float))
                powers.extend(get_arr(lines[5], float))
                keffs.extend(get_arr(lines[6], float))
                keffs_stddev.extend(get_arr(lines[7], float))
                tot_flux.extend(get_arr(lines[8], float))
            # Eat the lines without isos
            lines = lines[10:]
            # Now we can get the isotopes
            conc.append([])
            for iso in isos:
                # Find the line with our isotope
                for line in lines:
                    if line.startswith(iso):
                        conc[-1].append(get_arr(line, float))
                        break
        os.remove("mat_{}".format(csv_names[i]))

    # Now filter the timing results so we dont have multiple depletion end pts
    to_remove = []
    for i in range(len(op_labels) - 1):
        if op_labels[i] != op_labels[i + 1] and op_labels[i] == "deplete":
            to_remove.append(i)
    # But add back in the last point by removing it from the remove list
    if len(to_remove) > 0:
        del to_remove[-1]

    # Now do the filtering
    for i in reversed(to_remove):
        del case_labels[i]
        del op_labels[i]
        del step_idxs[i]
        del times[i]
        del powers[i]
        del keffs[i]
        del keffs_stddev[i]
        del tot_flux[i]
        for iso_idx in range(len(conc[0])):
            del conc[i][iso_idx]

    # Now we can get our data
    time = np.array(times)
    keff = np.zeros((len(time), 2))
    keff[:, 0] = keffs
    keff[:, 1] = keffs_stddev
    tot_flux = np.array(tot_flux)

    # Get the concentrationsin atom/b-cm
    N235 = np.zeros((len(time)))
    N134 = np.zeros((len(time)))
    N135 = np.zeros((len(time)))
    for t in range(len(time)):
        N235[t] = conc[0][0][t] * 1.E-24
        N134[t] = conc[0][1][t] * 1.E-24
        N135[t] = conc[0][2][t] * 1.E-24

    return time, keff, tot_flux, N235, N134, N135


def plot_compare(case_name, times, keff, phis, N235, N134, N135, calc_times,
                 calc_keff, calc_phis, calc_N235, calc_N134, calc_N135):
    plt.figure(figsize=(10, 8))
    plt.title("Keff vs Time")
    plt.plot(times, keff, label="Ref")
    plt.errorbar(calc_times, calc_keff[:, 0], yerr=calc_keff[:, 1],
                 label="ADDER")
    plt.legend(loc='best')
    plt.ylabel("Eigenvalue")
    plt.xlabel("Time [d]")
    plt.savefig(case_name + "_compare_keff.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title("Keff Bias vs Time")
    plt.plot(calc_times, get_error(times, keff, calc_times,
                                   calc_keff[:, 0], 1.e5))
    plt.ylabel("Eigenvalue Bias [pcm]")
    plt.xlabel("Time [d]")
    plt.savefig(case_name + "_compare_keff_bias.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title("Flux vs Time")
    plt.plot(times, phis, label="Ref Flux")
    plt.plot(calc_times, calc_phis, label="ADDER Flux")
    plt.legend(loc='best')
    plt.ylabel("Flux Level [n/cm^2-s]")
    plt.xlabel("Time [d]")
    plt.savefig(case_name + "_compare_phis.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title("Concentrations vs Time")
    plt.plot(times, N235, label="Ref $^{235}U$")
    plt.plot(times, N134, label="Ref $^{134}I$")
    plt.plot(calc_times, calc_N235, label="ADDER $^{235}U$")
    plt.plot(calc_times, calc_N134, label="ADDER $^{134}I$")
    plt.legend(loc='best')
    plt.ylabel("Concentrations [atom/b-cm]")
    plt.xlabel("Time [d]")
    plt.legend(loc='best')
    plt.savefig(case_name + "_compare_conc.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title("Concentrations vs Time")
    bias = get_error(times, N235, calc_times, calc_N235)
    plt.plot(calc_times, bias, label="$^{235}U$")
    bias = get_error(times, N134, calc_times, calc_N134)
    plt.plot(calc_times, bias, label="$^{134}I$")
    plt.legend(loc='best')
    plt.ylabel("Concentration Error [%]")
    plt.xlabel("Time [d]")
    plt.legend(loc='best')
    plt.savefig(case_name + "_compare_conc_error.png")
    plt.close()


class CouplingHarness(TestHarness):
    def __init__(self, depl_solver, depl_meth, num_adder_t,
                 test_lib_name="test.h5", input_fname="test.add"):
        super().__init__([], test_lib_name, input_fname)
        # Set the constants we will use for our solutions
        self.sig_f = 600E-24  # cm2
        self.sig_c = 50E-24  # cm2
        self.nu = 2.5    # n/fiss

        self.p0 = 400.   # MW
        # Use the Adder Q value for the data isotope, U-235
        self.q = 1.29927E-3 * 92. * 92. * np.sqrt(235.) + 33.12
        # This is in MeV, convert to MJ
        self.q *= 1.6021766208e-19
        self.v = 10. * 10. * 10.  # cc
        self.n235_0 = 2.0E24  # atom/b-cm
        self.n134_0 = 0.
        self.n135_0 = 0.

        # Do 4 yrs
        self.max_t = 4
        self.max_t *= 365. * 86400.   # sec
        self.num_t = int(self.max_t / 86400 * 24)  # one point per hr

        # file names
        self.mcnp_fname = "mcnp.inp"
        self.lib_name = "test"

        # Run parameters
        self.depl_solver = depl_solver
        self.depl_meth = depl_meth
        self.num_adder_t = num_adder_t

        # Get the reference solution
        times, keff, phis, N235, N134, N135 = \
            ref_soln(self.sig_f, self.sig_c, self.nu, self.p0, self.q, self.v,
                     self.n235_0, self.n134_0, self.n135_0, self.max_t,
                     self.num_t)
        self.ref_vals = {"time": times, "keff": keff, "phi": phis,
                         "N235": N235, "N134": N134, "N135": N135}

        # Initialize predictor and CECM first pass errors
        self.predictor_error = None
        self.cecm_error = None

    def _create_test_lib(self):
        depllib = DepletionLibrary(self.lib_name, np.array([0., 100.]))
        u235xs = ReactionData("cm2", 1)
        u235xs.add_type("fission", "cm2", np.array([self.sig_f]))
        u235nfy = YieldData()
        u235nfy.add_isotope("I134", 1.)
        depllib.add_isotope("U235", xs=u235xs, nfy=u235nfy)

        stable = DecayData(None, "s", 0.)
        I134xs = ReactionData("cm2", 1)
        I134xs.add_type("(n,gamma)", "cm2", np.array([self.sig_c]), "I135")
        depllib.add_isotope("I134", xs=I134xs, decay=stable)
        depllib.add_isotope("I135", decay=stable)
        depllib.set_isotope_indices()

        depllib.to_hdf5(self._test_lib_name)

    def _build_inputs(self):
        msg = \
            """case_name = 'Analytic IHM {} steps'
            neutronics_solver = "MCNP"
            neutronics_exec = {}
            num_threads = 1
            neutronics_library_file = ./xsdir
            neutronics_input_file = {}
            depletion_solver = {}
            depletion_exec = {}
            depletion_library_file = {}
            depletion_library_name = {}
            depletion_substeps = 1
            output_hdf5 = results.h5
            depletion_method = {}
            use_depletion_library_xs = True

            [operations]
                [[state 1]]
                    [[[deplete]]]
                        powers = {}
                        durations = {}
            """

        if self.depl_solver == "ORIGEN2.2":
            depl_exec = ORIGEN_EXEC
        else:
            depl_exec = ""

        # Compute the powers, and convert to comma-delimited string list
        powers_str = ", ".join(["{:1.3f}".format(self.p0)] * self.num_adder_t)

        # Compute the durations and convert to a comma-delimited string list in
        # units of days
        dt = (self.max_t / self.num_adder_t) / 86400.
        durations_str = ", ".join(["{:1.6f}".format(dt)] * self.num_adder_t)
        msg = msg.format(self.num_adder_t, MCNP_EXEC,
                         self.mcnp_fname, self.depl_solver,
                         depl_exec, self._test_lib_name, self.lib_name,
                         self.depl_meth, powers_str, durations_str)

        with open(self.input_fname, "w") as file:
            file.write(msg)

    def _compare_results(self):
        """Make sure the current results agree with the _true standard."""

        # Get the simulated predictor results and error from fine-bin
        # solution
        a_time, a_keff, a_phi, a_N235, a_N134, a_N135 = \
            ref_soln(self.sig_f, self.sig_c, self.nu, self.p0, self.q, self.v,
                     self.n235_0, self.n134_0, self.n135_0, self.max_t,
                     self.num_adder_t)
        # keff in units of pcm
        a_keff_err = get_error(self.ref_vals["time"], self.ref_vals["keff"],
                               a_time, a_keff, scaling=1.e5)
        # conc in units of percent
        a_N235_err = get_error(self.ref_vals["time"], self.ref_vals["N235"],
                               a_time, a_N235)
        a_N134_err = get_error(self.ref_vals["time"], self.ref_vals["N134"],
                               a_time, a_N134)
        a_N135_err = get_error(self.ref_vals["time"], self.ref_vals["N135"],
                               a_time, a_N135)

        # Get the ADDER solution to compare
        calc_time, calc_keff, calc_phi, calc_N235, calc_N134, calc_N135 = \
            get_adder_results()
        # Find keff and concentration errors vs time
        # keff in units of pcm
        keff_err = get_error(self.ref_vals["time"], self.ref_vals["keff"],
                             calc_time, calc_keff[:, 0], scaling=1.e5)
        # conc in units of percent
        N235_err = get_error(self.ref_vals["time"], self.ref_vals["N235"],
                             calc_time, calc_N235)
        N134_err = get_error(self.ref_vals["time"], self.ref_vals["N134"],
                             calc_time, calc_N134)
        N135_err = get_error(self.ref_vals["time"], self.ref_vals["N135"],
                             calc_time, calc_N135)

        if self.depl_meth == "predictor":
            # Ensure the analytic vs ADDER results within tolerance
            # All errors within 1.5% of analytic errors
            rtol = 0.015
            np.testing.assert_allclose(keff_err, a_keff_err, rtol=rtol)
            np.testing.assert_allclose(N235_err, a_N235_err, rtol=rtol)
            np.testing.assert_allclose(N134_err, a_N134_err, rtol=rtol)
            np.testing.assert_allclose(N135_err, a_N135_err, rtol=rtol)

            if self.predictor_error is None:
                # Store the error if this is our first time through
                self.predictor_error = \
                    np.array([self.num_adder_t, keff_err[-1], N235_err[-1],
                              N134_err[-1], N135_err[-1]])
            else:
                # Then make sure this error follows O(n)
                # Thus if we double the number of steps,
                # error should be half.
                n_ratio = self.predictor_error[0] / self.num_adder_t
                # Make sure all are within 20% of the expected error ratio
                new = np.array([keff_err[-1], N235_err[-1], N134_err[-1],
                                N135_err[-1]])
                old = self.predictor_error[1:]
                np.testing.assert_allclose(new / old, n_ratio, rtol=0.2)

        elif self.depl_meth == "cecm":
            # Ensure the CECM results are closer to truth than predictor
            assert keff_err[0] == a_keff_err[0]
            assert N235_err[0] == a_N235_err[0]
            assert N134_err[0] == a_N134_err[0]
            assert N135_err[0] == a_N135_err[0]
            np.testing.assert_array_less(np.abs(keff_err[1:]),
                                         np.abs(a_keff_err[1:]))
            np.testing.assert_array_less(np.abs(N235_err[1:]),
                                         np.abs(a_N235_err[1:]))
            np.testing.assert_array_less(np.abs(N134_err[1:]),
                                         np.abs(a_N134_err[1:]))
            np.testing.assert_array_less(np.abs(N135_err[1:]),
                                         np.abs(a_N135_err[1:]))

            if self.cecm_error is None:
                # Then save the results
                self.cecm_error = \
                    np.array([self.num_adder_t, keff_err[-1], N235_err[-1],
                              N134_err[-1], N135_err[-1]])
            else:
                # Make sure this error follows O(n^2)
                # Thus if we double the number of steps,
                # error should be quarter.
                n_ratio = self.cecm_error[0] / self.num_adder_t
                # Make sure all are within 20% of the expected error ratio
                new = np.array([keff_err[-1], N235_err[-1], N134_err[-1],
                                N135_err[-1]])
                old = self.cecm_error[1:]
                np.testing.assert_allclose(new / old, n_ratio**2, rtol=0.2)

        if config["plot"]:
            case_name = self.depl_meth + "_{}".format(self.num_adder_t)
            if self.depl_solver == "ORIGEN2.2":
                case_name += "orgn22_" + case_name
            else:
                case_name += self.depl_solver + "_" + case_name
            self.plot(case_name, calc_time, calc_keff, calc_phi, calc_N235,
                      calc_N134, calc_N135)

    def update_results(self):
        pass

    def execute_test(self):
        """Run ADDER with the appropriate arguments and check the outputs."""

        try:
            self._build_inputs()
            self._run_adder()
            self._compare_results()
        finally:
            self._cleanup()

    def _cleanup(self):
        """Delete statepoints, tally, and test files."""
        output = glob.glob('*.h5')
        output += glob.glob('case_*')
        output += ['adder.log', 'test.add']
        for f in output:
            if os.path.exists(f):
                os.remove(f)

    def plot(self, case_name, t, k, p, n235, n134, n135):
        plot_ref(case_name, self.ref_vals["time"], self.ref_vals["keff"],
                 self.ref_vals["phi"], self.ref_vals["N235"],
                 self.ref_vals["N134"], self.ref_vals["N135"])

        plot_compare(case_name, self.ref_vals["time"], self.ref_vals["keff"],
                     self.ref_vals["phi"], self.ref_vals["N235"],
                     self.ref_vals["N134"], self.ref_vals["N135"], t, k, p,
                     n235, n134, n135)
