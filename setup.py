from setuptools import setup

reqs = open('requirements.txt', 'r').read().strip().splitlines()

setup(
    name='Spectractor',
    version='1.1',
    packages=['spectractor', 'spectractor.extractor', 'spectractor.simulation', 'spectractor.fit'],
    install_requires=reqs,
    test_suite='nose.collector',
    tests_require=['nose'],
    package_dir={'spectractor': './spectractor'},
    package_data={'spectractor.extractor': ['dispersers/HoloPhAg/*.txt', 'dispersers/HoloPhP/*.txt',
                                           'dispersers/HoloAmAg/*.txt', 'dispersers/Thor300/*.txt',
                                           'dispersers/Ron200/*.txt', 'dispersers/Ron400/*.txt'],
                  'spectractor.simulation': ['CTIOThroughput/*.txt']},
    url='https://github.com/LSSTDESC/Spectractor',
    license='',
    author='J. Neveu, S. Dagoret-Campagne',
    author_email='jneveu@lal.in2p3.fr',
    description='',
)
