from setuptools import setup
import os
import re

reqs = []


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append( os.path.join("..", path, filename))
    return paths


disperser_files = package_files('./spectractor/extractor/dispersers')


# skip using requirements.txt in a CONDA environment
if os.getenv('CONDA_PREFIX') is None:
    reqs = open('requirements.txt', 'r').read().strip().splitlines()

if os.getenv('READTHEDOCS') and 'mpi4py' in reqs:
    reqs.remove('mpi4py')

with open('README.md') as file:
    long_description = file.read()

# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file = os.path.join('spectractor', '_version.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    current_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print(f'Spectractor version is {current_version}')

setup(
    name='Spectractor',
    version=current_version,
    packages=['spectractor', 'spectractor.extractor', 'spectractor.simulation', 'spectractor.fit'],
    install_requires=reqs,
    test_suite='nose.collector',
    tests_require=['nose'],
    package_dir={'spectractor': './spectractor'},
    package_data={'': disperser_files,
                  'spectractor': ['../config/*.ini'],
                  'spectractor.simulation': ['CTIOThroughput/*.txt', 'AuxTelThroughput/*.txt', 'StarDiceThroughput/*.txt']},
    url='https://github.com/LSSTDESC/Spectractor',
    license='BSD',
    python_requires='>=3.7',
    author='J. Neveu, S. Dagoret-Campagne',
    author_email='jeremy.neveu@universite-paris-saclay.fr',
    description='',
    long_description=long_description,
)
