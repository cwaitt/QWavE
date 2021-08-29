"""
SE_Solver
A SE solver for simple models such as a particle in a box and complicated models for an arbitrary potential
"""

# Add imports here
from .schrodinger import *
from .partition_func import *
from .thermo import *
from .statistics import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
