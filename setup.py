from setuptools import setup, find_packages

from ParProcCo import __version__

setup(
    name='ParProcCo',
    version=__version__,
    description='Parallel Processing Coordinator. Splits dataset processing to run parallel cluster jobs and aggregates outputs',
    author_email="dataanalysis@diamond.ac.uk",
    packages=find_packages(),
    install_requires=['h5py', 'numpy', 'PyYAML', 'uge-drmaa2'],
    extras_require={'testing': ['parameterized'],},
    scripts=['scripts/b24_cluster_submit','scripts/bl_cluster_submit', 'scripts/i07_cluster_submit',
             'scripts/i16_cluster_submit', 'scripts/nxdata_aggregate', 'scripts/p99_cluster_submit',
             'scripts/ppc_cluster_runner', 'scripts/ppc_cluster_submit'],
    url='https://github.com/DiamondLightSource/ParProcCo',
)
