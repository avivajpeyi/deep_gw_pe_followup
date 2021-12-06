import bilby
import numpy as np

from deep_gw_pe_followup.restricted_prior.prior import RestrictedPrior
from deep_gw_pe_followup.restricted_source_model import \
    lal_binary_black_hole_qxeff_restricted

DURATION = 4.
SAMPLING_FREQ = 2048.
MIN_FREQ = 20

OUTDIR = 'outdir'
LABEL = 'fast_tutorial'

bilby.core.utils.setup_logger(outdir=OUTDIR, label=LABEL)
logger = bilby.core.utils.logger

np.random.seed(0)

INJECTION_PARAM = {'a_1': 0.002029993860587983, 'cos_tilt_1': 0.30398947450543734,
                   'cos_tilt_2': 0.9088301398360985, 'a_2': 0.6567943667372238, 'mass_ratio': 0.2, 'chi_eff': 0.1,
                   'phi_12': 5.838675984634243, 'phi_jl': 4.9905128637161456, 'chirp_mass': 40.62153505079806,
                   'luminosity_distance': 250, 'dec': 0.3261767928551774, 'ra': 2.6694652170836255,
                   'theta_jn': 1.5324822247502017, 'psi': 2.7544433486044135, 'phase': 1.5585445420281385,
                   'geocent_time': 0}


def setup_liklihood(injection_parameters):
    logger.info("Setting up likelihood")

    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=50.,
                              minimum_frequency=MIN_FREQ)

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQ,
        frequency_domain_source_model=lal_binary_black_hole_qxeff_restricted,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments)

    ifos = bilby.gw.detector.InterferometerList(['H1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLING_FREQ, duration=DURATION,
        start_time=injection_parameters['geocent_time'] - 3)
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection_parameters)

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator)
    return likelihood


def setup_priors(injection_parameters):
    logger.info("Setting up prior")
    priors = RestrictedPrior(filename="restricted.prior")
    for param in ['geocent_time', 'phi_12', 'phi_jl', 'luminosity_distance', 'dec', 'ra', 'theta_jn', 'psi', 'phase']:
        priors[param] = injection_parameters[param]
    priors.time_prior()
    return priors


def main():
    likelihood = setup_liklihood(INJECTION_PARAM)
    priors = setup_priors(INJECTION_PARAM)

    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=512,
        injection_parameters=INJECTION_PARAM, outdir=OUTDIR, label=LABEL, npool=1)

    # Make a corner plot.
    result.plot_corner()


if __name__ == '__main__':
    main()
