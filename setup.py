# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import re

from setuptools import setup, find_packages
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.install import install as _install

VERSION_PY = """
# This file is originally generated from Git information by running 'setup.py
# version'. Distribution tarballs contain a pre-generated copy of this file.

__version__ = '%s'
"""


def update_version_py():
    if not os.path.isdir(".git"):
        print("This does not appear to be a Git repository.")
        return
    try:
        p = subprocess.Popen(["git", "describe",
                              "--tags", "--always"],
                             stdout=subprocess.PIPE)
    except EnvironmentError:
        print("unable to run git, leaving hicmatrix/_version.py alone")
        return
    stdout = p.communicate()[0]
    if p.returncode != 0:
        print("unable to run git, leaving hicmatrix/_version.py alone")
        return
    ver = stdout.strip()
    f = open("hicmatrix/_version.py", "w")
    f.write(VERSION_PY % ver)
    f.close()
    print("set hicmatrix/_version.py to '%s'" % ver)


def get_version():
    try:
        f = open("hicmatrix/_version.py")
    except EnvironmentError:
        return None
    for line in f.readlines():
        mo = re.match("__version__ = '([^']+)'", line)
        if mo:
            ver = mo.group(1)
            return ver
    return None


class sdist(_sdist):

    def run(self):
        # update_version_py()
        self.distribution.metadata.version = get_version()
        return _sdist.run(self)

# Install class to check for external dependencies from OS environment


class install(_install):

    def run(self):
        # update_version_py()
        self.distribution.metadata.version = get_version()
        _install.run(self)
        return

    def checkProgramIsInstalled(self, program, args, where_to_download,
                                affected_tools):
        try:
            subprocess.Popen([program, args],
                             stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE)
            return True
        except EnvironmentError:
            # handle file not found error.
            # the config file is installed in:
            msg = "\n**{0} not found. This " \
                  "program is needed for the following "\
                  "tools to work properly:\n"\
                  " {1}\n"\
                  "{0} can be downloaded from here:\n " \
                  " {2}\n".format(program, affected_tools,
                                  where_to_download)
            sys.stderr.write(msg)

        except Exception as e:
            sys.stderr.write("Error: {}".format(e))


install_requires_py = ["numpy >= 1.16.*",
                       "scipy >= 1.2.*",
                       "tables >= 3.5.*",
                       "pandas >= 0.25.*",
                       "cooler >= 0.8.9",
                       "intervaltree >= 3.0.*"
                       ]

setup(
    name='HiCMatrix',
    version=get_version(),
    author='Joachim Wolff, Leily Rabbani, Vivek Bhardwaj, Fidel Ramirez',
    author_email='wolffj@informatik.uni-freiburg.de',
    packages=find_packages(),
    include_package_data=True,
    package_dir={'hicmatrix': 'hicmatrix'},
    url='https://github.com/deeptools/HiCMatrix',
    license='LICENSE',
    description='Helper package which implements HiCMatrix class for HiCExplorer, pyGenomeTracks and scHiCExplorer.',
    long_description=open('README.rst').read(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
    install_requires=install_requires_py,
    zip_safe=False,
    cmdclass={'sdist': sdist, 'install': install}
)
