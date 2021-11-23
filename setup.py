from setuptools import setup

from ParProcCo import __version__

setup(
    name='ParProcCo',
    version=__version__,
    description='Parallel Processing Coordinator. Splits dataset processing to run parallel cluster jobs and aggregates outputs',
    packages=['ParProcCo'],
    install_requires=['drmaa2', 'h5py', 'numpy', 'parameterized', 'PyYAML'],
)
