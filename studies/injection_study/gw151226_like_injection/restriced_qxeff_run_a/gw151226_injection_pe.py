import bilby
import numpy as np

# from deep_gw_pe_followup.restricted_prior.prior import RestrictedPrior
# from deep_gw_pe_followup.restricted_source_model import \
#     lal_binary_black_hole_qxeff_restricted

from LVK_GW151226_MAP_param import PARAM as injection_parameters

logger = bilby.core.utils.logger
OUTDIR = 'outdir'
LABEL = 'GW151226_like_injection'
bilby.core.utils.setup_logger(outdir=OUTDIR, label=LABEL)

np.random.seed(0)


def setup_lnl():
    duration = 8.
    sampling_frequency = 2048.
    minimum_frequency = 20

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

    logger.info("Setting up likelihood")
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator
    )

    return likelihood


def setup_prior():
    # priors = RestrictedPrior("restricted.prior")
    # priors['geocent_time'] = injection_parameters['geocent_time']
    # priors.time_prior()
    # return priors
    priors =


def run_sampler(likelihood, priors):
    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=512,
        injection_parameters=injection_parameters, outdir=OUTDIR, label=LABEL, npool=1)

    # Make a corner plot.x
    result.plot_corner()

def main():
    likelihood = setup_lnl()
    priors = setup_prior()
    run_sampler(likelihood, priors)

if __name__ == '__main__':
    main()