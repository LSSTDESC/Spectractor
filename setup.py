from setuptools import setup

setup(
    name='Spectractor',
    version='1.1',
    packages=['spectractor', 'spectractor.pipeline', 'spectractor.simulation', 'spectractor.fit'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'astroquery',
                      'coloredlogs', 'scikit-image', 'pysynphot', 'emcee'],
    test_suite='nose.collector',
    tests_require=['nose'],
    package_dir={'spectractor': './spectractor'},
    package_data={'spectractor.pipeline': ['dispersers/HoloPhAg/*.txt', 'dispersers/HoloPhP/*.txt',
                                           'dispersers/HoloAmAg/*.txt', 'dispersers/Thor300/*.txt',
                                           'dispersers/Ron200/*.txt', 'dispersers/Ron400/*.txt'],
                  'spectractor.simulation': ['CTIOThroughput/*.txt']},
    url='https://github.com/LSSTDESC/Spectractor',
    license='',
    author='J. Neveu, S. Dagoret-Campagne',
    author_email='jneveu@lal.in2p3.fr',
    description=''
)
