from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'Jakob Kienegger'
LONG_DESCRIPTION = 'Social Force Speaker Trajectories'

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

setup(
        name="sfm",
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages()
)
