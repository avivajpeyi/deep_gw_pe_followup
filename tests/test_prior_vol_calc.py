import unittest
from deep_gw_pe_followup.restricted_prior.prior_vol_calc import calc_prior_vol
from bilby.core.prior import PriorDict, Uniform

class TestPriorVolCalc(unittest.TestCase):
    def test_prior_vol_calc(self):
        simple_prior = PriorDict(dict(
            a=Uniform(0,5),
        ))
        tru_prior_vol = simple_prior['a'].maximum-simple_prior['a'].minimum
        numerical_prior_vol = calc_prior_vol(simple_prior)
        self.assertAlmostEqual(tru_prior_vol, numerical_prior_vol)


if __name__ == '__main__':
    unittest.main()
