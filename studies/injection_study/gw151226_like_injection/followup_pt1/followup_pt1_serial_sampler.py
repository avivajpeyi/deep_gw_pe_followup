import bilby
import numpy as np

from deep_gw_pe_followup.restricted_prior.prior import RestrictedPrior
from deep_gw_pe_followup.restricted_source_model import \
    lal_binary_black_hole_qxeff_restricted

DURATION = 8
SAMPLING_FREQ = 2048.
MIN_FREQ = 20

OUTDIR = 'outdir_serial'
LABEL = 'serial'

bilby.core.utils.setup_logger(outdir=OUTDIR, label=LABEL)
logger = bilby.core.utils.logger

np.random.seed(0)

INJECTION_PARAM = {'luminosity_distance': 800, 'mass_ratio': 0.26512986754630535, 'chirp_mass': 9.702165419247658, 'a_1': 0.9873737113225366, 'a_2': 0.8435528737659149, 'tilt_1': 1.0401383271620692, 'tilt_2': 2.5427989903091914, 'phi_12': 1.695426569783911, 'phi_jl': 3.3438963165263136, 'dec': 0.33364118287594846, 'ra': 0.6738167456098734, 'cos_theta_jn': -0.8684204854784665, 'psi': 2.06954952383973, 'phase': 0.8866172464617089, 'geocent_time': 0, 'theta_jn': 2.6228040981756555, 'chi_eff':0.24896368}


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

    ifos = bilby.gw.detector.InterferometerList(['H1', "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLING_FREQ, duration=DURATION,
        start_time=injection_parameters['geocent_time'] - (DURATION -1))
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection_parameters)

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator)
    return likelihood


def setup_priors(injection_parameters):
    logger.info("Setting up prior")
    priors = RestrictedPrior("followup_pt1.prior")
    priors.plot_cache()
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
