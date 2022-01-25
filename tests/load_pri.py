from deep_gw_pe_followup.restricted_prior import RestrictedPrior

import unittest


class TestPriorLoad(unittest.TestCase):
    def test_load(self):
        RestrictedPrior(
            filename="../studies/injection_study/gw151226_like_injection/followup_pt1/followup_pt1.prior",
            mcmc_n=5
        )
