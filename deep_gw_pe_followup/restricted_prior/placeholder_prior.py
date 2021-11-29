from bilby.core.prior import DeltaFunction, Prior
import numpy as np


class PlaceholderPrior(Prior):
    pass


class PlaceholderDelta(Prior):

    def __init__(self, peak, name=None, latex_label=None, unit=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit,
                         minimum=peak, maximum=peak, check_range_nonzero=False)
        self.peak = peak
        self._is_fixed = True

    def rescale(self, val):
        return self.peak * val ** 0

    def prob(self, val):
        at_peak = (val == self.peak)
        return np.nan_to_num(np.multiply(at_peak, np.inf))

    def cdf(self, val):
        return np.ones_like(val) * (val > self.peak)
