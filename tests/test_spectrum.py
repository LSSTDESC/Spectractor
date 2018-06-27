from unittest import TestCase
from spectractor.pipeline.spectroscopy import Spectrum
import spectractor.parameters as parameters
import numpy as np


class TestSpectrum(TestCase):
    def test_convert_from_ADUrate_to_flam(self):
        s = Spectrum()
        s.lambdas = np.linspace(parameters.LAMBDA_MIN,parameters.LAMBDA_MAX,100)
        s.lambdas_binwidths = np.gradient(s.lambdas)
        s.data = np.ones_like(s.lambdas)
        s.convert_from_ADUrate_to_flam()
        assert np.max(s.data) < 1e-14

    def test_convert_from_flam_to_ADUrate(self):
        s = Spectrum()
        s.lambdas = np.linspace(parameters.LAMBDA_MIN,parameters.LAMBDA_MAX,100)
        s.lambdas_binwidths = np.gradient(s.lambdas)
        s.data = np.ones_like(s.lambdas)
        s.convert_from_flam_to_ADUrate()
        assert np.max(s.data) > 1e14

    def test_load_filter(self):
        s = Spectrum()
        s.filter = 'FGB37'
        s.load_filter()
        assert parameters.LAMBDA_MIN == parameters.FGB37['min']
        assert parameters.LAMBDA_MAX == parameters.FGB37['max']
