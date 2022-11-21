#!/usr/bin/env python
import glob
import sys
from setuptools import setup, find_packages


# Determine shared library suffix
if sys.platform == 'darwin':
    suffix = 'dylib'
else:
    suffix = 'so'

# Get version information from __init__.py. This is ugly, but more
# reliable than using an import.
with open('adder/__init__.py', 'r') as f:
    version = f.readlines()[-1].split()[-1].strip("'")

kwargs = {
    'name': 'ADDER',
    'version': version,
    'packages': find_packages(exclude=['tests*']),
    'scripts': glob.glob('scripts/adder_*'),

    # Metadata
    'author': 'Adam Nelson',
    'author_email': 'agnelson@anl.gov',
    'description': 'ADDER',
    'url': 'https://svn.inside.anl.gov/repos/adder/',
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],

    # Required dependencies
    'install_requires': [
        'numpy', 'h5py', "configobj", "scipy", "pytest"
    ],

    # Optional dependencies
    'extras_require': {
        'test': ['pytest-cov', 'matplotlib'],
        'mcnp': ['mcnptools']
    },

    # Data files and libraries
    'package_data': {
        'adder': ['mass16.txt']
    },

    # Set the main executable as a script
    'entry_points': {
        'console_scripts': [
            'adder = adder.__main__:main'
        ]
    }
}


setup(**kwargs)
