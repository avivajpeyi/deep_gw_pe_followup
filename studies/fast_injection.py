import bilby
import numpy as np

from deep_gw_pe_followup.restricted_prior.prior import RestrictedPrior
from deep_gw_pe_followup.restricted_source_model import \
    lal_binary_black_hole_qxeff_restricted

import datetime

duration = 4.
sampling_frequency = 2048.
minimum_frequency = 20

outdir = 'outdir'
label = 'fast_tutorial'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

logger = bilby.core.utils.logger

np.random.seed(0)

injection_parameters = {'a_1': 0.002029993860587983, 'cos_tilt_1': 0.30398947450543734,
                        'cos_tilt_2': 0.9088301398360985, 'a_2': 0.6567943667372238, 'mass_ratio': 0.2, 'chi_eff': 0.1,
                        'phi_12': 5.838675984634243, 'phi_jl': 4.9905128637161456, 'chirp_mass': 40.62153505079806,
                        'luminosity_distance': 250, 'dec': 0.3261767928551774, 'ra': 2.6694652170836255,
                        'theta_jn': 1.5324822247502017, 'psi': 2.7544433486044135, 'phase': 1.5585445420281385, 'geocent_time':0}

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50.,
                          minimum_frequency=minimum_frequency)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=lal_binary_black_hole_qxeff_restricted,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

priors = RestrictedPrior("restricted.prior")
priors['geocent_time'] = injection_parameters['geocent_time']
# priors.validate_prior(duration, minimum_frequency)


logger.info("Setting up likelihood")
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)


def _time_likelihood(self, n_evaluations=100):
    """ Times the likelihood evaluation and print an info message

    Parameters
    ==========
    n_evaluations: int
        The number of evaluations to estimate the evaluation time from

    """

    t1 = datetime.datetime.now()
    for _ in range(n_evaluations):
        theta = self.priors.sample_subset_constrained_as_array(
            self._search_parameter_keys, size=1)[:, 0]
        self.log_likelihood(theta)
    total_time = (datetime.datetime.now() - t1).total_seconds()
    self._log_likelihood_eval_time = total_time / n_evaluations

    if self._log_likelihood_eval_time == 0:
        self._log_likelihood_eval_time = np.nan
        logger.info("Unable to measure single likelihood time")
    else:
        logger.info("Single likelihood evaluation took {:.3e} s"
                    .format(self._log_likelihood_eval_time))


try:
    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
        injection_parameters=injection_parameters, outdir=outdir, label=label)
except Exception as e:
    print(e)
    print("wait!")


# Make a corner plot.
result.plot_corner()
