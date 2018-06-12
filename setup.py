from setuptools import setup

setup(
    name='Spectractor',
    version='1.0',
    packages=['spectractor'],
    install_requires=['numpy','astropy','astroquery','coloredlogs','scikit-image'],
    test_suite='nose.collector',
    tests_require=['nose'],
    url='https://github.com/LSSTDESC/Spectractor',
    license='',
    author='J. Neveu, S. Dagoret-Campagne',
    author_email='jneveu@lal.in2p3.fr',
    description=''
)
