#/urs/bin/env python

__version__ = "0.0.1-dev"

import os
from distutils.core import setup

classes = """
    Development Status :: 1 - Planning
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Statitics
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.4
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""

setup(name='monte-carlo-power',
      version="0.0.1-dev",
      license='BSD2',
      description="Library for testing monte carlo effect sizes",
      long_description=("Library for testing monte carlo effect sizes"),
      author="J W Debelius",
      author_email="j.debelius@gmail.com",
      maintainer="J W Debelius",
      maintainer_email="j.debelius@gmail.com",
      packages=['emp_power', 'emp_power.tests'],
      )
