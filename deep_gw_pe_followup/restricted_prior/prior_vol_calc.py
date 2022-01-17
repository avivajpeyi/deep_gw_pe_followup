from bilby import run_sampler
from bilby.core.prior import PriorDict, Uniform
from bilby.core.likelihood import Likelihood
import numpy as np


class ZeroLikelihood(Likelihood):
    def log_likelihood(self):
        return 0


def calc_prior_vol(prior):
    likelihood = ZeroLikelihood(parameters=prior.sample(1))
    result = run_sampler(likelihood=likelihood, priors=prior, save=False, verbose=0, clean=True)
    return np.exp(result.log_evidence)


def test():
    simple_prior = PriorDict({'a': Uniform(0, 5)})
    tru_prior_vol = simple_prior['a'].maximum - simple_prior['a'].minimum
    numerical_prior_vol = calc_prior_vol(simple_prior)
    assert np.isclose(numerical_prior_vol, tru_prior_vol), f"Numerical: {numerical_prior_vol}, True:{tru_prior_vol}"


if __name__ == '__main__':
    test()
