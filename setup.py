from setuptools import setup, find_packages

from ParProcCo import __version__

setup(
    name='ParProcCo',
    version=__version__,
    description='Parallel Processing Coordinator. Splits dataset processing to run parallel cluster jobs and aggregates outputs',
    author_email="dataanalysis@diamond.ac.uk",
    packages=find_packages(),
    install_requires=['h5py', 'numpy', 'PyYAML', 'uge-drmaa2'],
    extras_require=['parameterized'],
    url='https://github.com/DiamondLightSource/ParProcCo',
)
