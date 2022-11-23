==================================================================
Advanced Dimensional Depletion for Engineering of Reactors (ADDER)
==================================================================

The Advanced Dimensional Depletion for Engineering of Reactors (ADDER) software
is a flexible, performant, and modern fuel management and depletion tool. The
target audience of this tool is first the Research and Test Reactor Program
at Argonne National Laboratory (ANL) and, eventually, to experts outside of ANL.
This report is the user guide for the software release referred to as ADDER
v1.0.1.

This software is fundamentally an interface between the user, external neutron
diffusion or transport theory solvers, and a depletion solver. The user will
define the reactor with the necessary input to the neutron diffusion/transport
solver, and ADDER will provide the user with the ability to deplete the reactor
for a given power history, shuffle fuel in the core and load or
remove fuel from the core, perform criticality searches, and stochastic volume
computations. ADDER will eventually be able to perform other analyses necessary
for the design of a reactor, such as branch calculations for multiple xenon
conditions or operating temperatures.

The initial application of ADDER will be to utilize the MCNP6_ (or MCNP5_)
software for neutron transport and either ORIGEN2_ for depletion and decay
functionality or an internal Chebyshev rational approximation method (CRAM_)
solver.

This README file provides a primer for the ADDER software. The complete 
software manual is located in the ``Documents`` folder of this repository.

****************
Installing ADDER
****************

ADDER is written in the Python 3 programming language. It therefore requires
that a Python 3.7-compatible interpreter be installed locally.

In addition to Python itself, ADDER relies on third-party packages.
All prerequisites can be installed using Anaconda_ (recommended), Python's
pip_ command, or through the package manager in most Linux distributions.

.. admonition:: Required
   :class: error

   `NumPy <http://www.numpy.org/>`_
      NumPy is used extensively within the ADDER for its powerful N-dimensional
      array.

   `SciPy <https://www.scipy.org/>`_
      SciPy's sparse linear algebra capabilities are used by ADDER's CRAM
      solver.

   `h5py <http://www.h5py.org/>`_
      h5py provides Python bindings to the HDF5-formatted Depletion data
      library. The h5py package is needed to provide access to data within these
      files from Python.

   `Configobj <https://configobj.readthedocs.io/en/stable/configobj.htm>`_
      Configobj is a simple but powerful configuration file reader with syntax
      similar to the classic Windows .INI file format.

.. admonition:: Optional
   :class: note

   `pytest <https://docs.pytest.org>`_
      The pytest framework is used for unit testing the ADDER code.

   `pytest-cov <https://github.com/pytest-dev/pytest-cov>`_
      The pytest-cov package is a plugin to pytest which reports code coverage
      of the ADDER test suite.

   `Matplotlib <http://matplotlib.org/>`_
      Matplotlib is used to plot certain results for testing purposes.

.. _Anaconda: https://conda.io/docs/
.. _pip: https://pip.pypa.io/en/stable/

These prerequisites will be installed automatically from the pip_ repository
when installing ADDER if they are not available already on the base Operating
System or Anaconda_ environment.

The installation of ADDER is trivially performed by executing the following
command from the root directory of the ADDER distribution/repository:

.. code-block:: sh

    pip install .

To install ADDER for a single user, the following command should be executed
by that user from the root directory of the ADDER distribution/repository:

.. code-block:: sh

    pip install --user .

The ADDER executable, ``adder`` can now be executed in the working directory
of interest.

.. note::
   If the command to execute Python 3 on the installation system is
   ``python3`` instead of ``python``, then ``pip`` in the above installation
   steps should be replaced with ``pip3``.

***************
Executing ADDER
***************
ADDER can be executed in whichever directory the user wishes. The simplest
usage of ADDER includes providing one input argument, the name of the
already-generated ADDER input file. The format of this input file is described
in the Input File Format section. Input flags are discussed in the block below.

In addition to the ADDER input file, an HDF5_ file containing the depletion
data library is utilized. The name of this file is specified by the
:ref:`depletion_library_file <depletion_library_file>` parameter in the ADDER
input file. The depletion data library file format is described in
:ref:`library_format`.

When executed, ADDER creates the following output files:

1. ``adder.log``: This is a continually-updated log of all messages from ADDER.
   Most messages in this file are echoes of the same information displayed to
   screen, however the ``adder.log`` file also includes additional information
   such as the configuration settings after validation and initial processing.

2. ``results.h5`` or an output file name as described by the
   :ref:`output_hdf5 <output_hdf5>` parameter in the input file: This file
   contains all results from the execution including the material inventories
   at each state point, the neutronics solver output files, etc.

3. Automatically-generated neutronics solver input and output files: ADDER
   leaves the neutronics solver input and output files generated during ADDER
   execution for easy verification and progress monitoring.

ADDER offers the following command line flags:

1. ``-n`` or ``--no-deplete``: This flag instructs ADDER to perform fuel
   management operations, but does not execute the neutronics or depletion
   solvers. This is used to produce typical fuel management output and logs
   which can be used to interrogate the names ADDER has applied to copied
   objects.

2. ``-f`` or ``--fast-forward``: This flag instructs ADDER to not execute
   neutronics calculations performed if the output files from previous ADDER
   runs are in the current working directory. When using this option, the user
   must be sure that the neutronics output files are consistent with the
   current case. Note that this only skips the neutronics calculations
   that are performed as part of deplete and geom_sweep operation blocks. This
   flag can be useful as a sort of restart capability, allowing the most costly
   computations to be skipped if they have already been performed in a previous
   ADDER execution.

3. ``-h`` or ``--help``: This flag prints the above options and descriptions.

Finally, when using depletion solvers that must be interacted with via textual
input and output files (currently only ORIGEN2), these inputs and outputs will
be placed according to the following hierarchical rules:

1.  The directory named by the TMPDIR environment variable.

2.  The directory named by the TEMP environment variable.

3.  The directory named by the TMP environment variable.

4.  A platform-specific location:

    a. On Windows, the directories C:\TEMP, C:\TMP, \TEMP, and \TMP, in that order.

    b. On all other platforms, the directories /tmp, /var/tmp, and /usr/tmp, in that order.

5. As a last resort, the current working directory.

Understanding this hierarchy is useful for utilizing fast I/O resources such as
RAM-based or local filesystems as available. It should be noted that if the
depletion solver fails to execute, ADDER will copy the working directory to
ADDER's main execution directory for later user inspection.


******************
ADDER Data Sources
******************

Depletion solvers require knowledge of the transmutation chain, the reaction
rate cross-sections, decay rates, and branching ratios. These are all defined
for ADDER in a depletion library HDF5 file (discussed in the Depletion Data
Library section). This library can include either one-group or multigroup
cross-section data. These cross-sections can either be used directly or the
neutronics solver can be used to directly obtain the corresponding reaction
rates in each depleting material. The former will lead to faster neutronics
computations (with Monte Carlo solvers) while the latter will lead to a more
accurate solution. This is controlled with the ``use_depletion_library_xs``
flag discussed later.

For simplicity of user input for feed and removal rates for molten-salt reactor
(MSR) analyses, ADDER also uses isotopic natural abundances to expand
user-provided elements. The natural abundances for this data are stored in the
``adder/data.py`` module. This data is sourced from the 2013 IUPAC_ Technical
Report.

.. note::
   If different natural abundance data is desired, the user must modify the
   ``adder/data.py`` file directly.

Next, atomic mass data is used internally by ADDER to convert any material
concentrations provided in the neutronics input from weight percent to atom
percent. This data is sourced directly from the ``adder/mass16.txt`` file.
This file is an official ASCII reformatting of the Atomic Mass Evaluation 2016
(AME2016_) data from the IAEA's Nuclear Data Service. The official file can be
found at the mass16_ reference. If the user wishes to modify this data, they
simply need to modify/overwrite the ``adder/mass16.txt`` file.

.. note::
   If different atomic mass data is desired, the user must modify the
   ``adder/mass16.txt`` file directly.

***************
Utility Scripts
***************

The ADDER software is distributed with a few utility scripts, included in the
``scripts`` folder:

* ``adder_convert_origen22_rsicc_libraries.py``: 
	this script converts the entire suite of libraries distributed with the 
	Radiation Safety Information Computational Center (RSICC) distribution 
	of ORIGEN2.2. The path to the library folder needs to be provided via 
	the command-line argument ``-r``. Additional information can be found 
	in Section 4.1.1 of the manual.
* ``adder_convert_origen22_library.py``: 
	this scripts allows users to convert an individual ORIGEN2.2 library file, 
	containing the desired cross-sections and yield values to convert. 
	The script requires several arguments. More information can be found in 
	Section 4.1.1 of the manual.
* ``adder_extract_materials.py``: 
	extracts the ADDER HDF5 output (``results.h5``) and generates a ``.csv`` 
	file containing the power, k:eff:, one-group fluxes, Q-recoverable energy, 
	and isotopic data for a selected material. The script requires 
	(in the following order): the path to the ``results.h5`` file generated
	by ADDER, the path to the desired output ``.csv`` file, and the name of the 
	required material. Additional information can be found in Section 3.1 of the manual.


********
Contacts
********
For inquiries about the ADDER software please reach out to our development team at adder@anl.gov. 

************
Citing ADDER
************
If you use ADDER for your work, please cite the following reference:
Anderson, K., Mascolino, V., and Nelson, A. G.. 2022. "User Guide to the 
Advanced Dimensional Depletion for Engineering of Reactors (ADDER) Software". 
United States. <https://doi.org/10.2172/1866062>. <https://www.osti.gov/servlets/purl/1866062>.

**********
References
**********

.. _ORIGEN2:

A.G. Croff, A User's Manual for the ORIGEN2 Computer Code, ORNL/TM-7175,
Oak Ridge National Laboratory, Oak Ridge, USA (1980).

.. _MCNP6:

C.J. Werner (editor), "MCNP User's Manual - Code Version 6.2", LA-UR-17-19981,
Los Alamos National Laboratory, Los Alamos, USA (2017).

.. _MCNP5:

X-5 Monte Carlo Team, MCNP – A General Monte Carlo N-Particle Transport
Code, Version 5, Volume I (LA-UR-03-1987), Volume II (LA-CP-03-0245),
Volume III (LA-CP-03-0284), Los Alamos National Laboratory, Los Alamos,
USA (2003).

.. _CRAM:

M. Pusa, "Higher-Order Chebyshev Rational Approximation Method and Application
to Burnup Equations", Nucl. Sci. Eng., 182:3, 297-318 (2016).

.. _IUPAC:

J. Meija, et al., “Isotopic Compositions of the Elements 2013 (IUPAC Technical
Report)”, Pure Appl. Chem., 88:3, 293-306 (2016).

.. _AME2016:

W.J. Huang, G. Audi, M. Wang, F.G. Kondev, S. Naimi and X. Xu, “The AME2016
Atomic Mass Evaluation (I)”, Chinese Physics C, 41:3 03002 (2017)..

.. _mass16: https://www-nds.iaea.org/amdc/ame2016/mass16.txt

.. _HDF5: http://www.hdfgroup.org/HDF5/
