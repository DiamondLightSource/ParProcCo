from setuptools import setup

from ParProcCo import __version__

setup(
    name="ParProcCo",
    version=__version__,
    description="Parallel Processing Coordinator. Splits dataset processing to run parallel cluster jobs and aggregates outputs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author_email="dataanalysis@diamond.ac.uk",
    packages=["ParProcCo", "ParProcCo.slurm", "ParProcCo.test"],
    install_requires=["h5py", "pydantic", "pyyaml", "requests"],
    extras_require={
        "testing": ["parameterized", "pytest"],
        "dev": ["datamodel-code-generator"],
    },
    scripts=[
        "scripts/nxdata_aggregate",
        "scripts/ppc_cluster_runner",
        "scripts/ppc_cluster_submit",
    ],
    url="https://github.com/DiamondLightSource/ParProcCo",
    python_requires=">=3.10",
)
