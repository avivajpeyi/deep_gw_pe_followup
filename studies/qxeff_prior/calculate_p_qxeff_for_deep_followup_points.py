from GW151226_reader.data_reader import load_res

from bilby.gw.prior import PriorDict, Uniform
from deep_gw_pe_followup.sample_cacher.kde2d import get_kde
from deep_gw_pe_followup.sample_cacher import cacher
from effective_spins.priors_conditional_on_xeff import p_param_and_xeff
import numpy as np

SAMPLES_FILE = "qxeff_samples.h5"
ASTRO_SAMPLES_FILE = "astro_qxeff_samples.h5"


NUMERICAL_QXEFF_NORM_FACTOR = 5280.323631278067

LOW_Q = {'q': 0.15, 'xeff': 0.5}
HIGH_Q = {'q': 0.68, 'xeff': 0.15}

def get_max_l_data():
    return dict(
        LOW_Q = {'q': 0.15, 'xeff': 0.5},
        HIGH_Q = {'q': 0.68, 'xeff': 0.15}
    )


def get_max_l_data_raw():
    res = load_res()
    for l, r in res.items():
        max_param = r.get_max_log_likelihood_param()
        print(l, dict(q=max_param["mass_ratio"], xeff=max_param["chi_eff"]))

    keys = ["chi_eff", "mass_ratio"]
    lvk_max_param = {
        k: v for k, v in res["lvk_data"].get_max_log_likelihood_param().items() if k in keys
    }
    ias_max_param = {
        k: v for k, v in res["ias_data"].get_max_log_likelihood_param().items() if k in keys
    }
    return dict(LVK=lvk_max_param, IAS=ias_max_param)

def get_kde_object(samp_file):
    samp_df = cacher.load_probabilities(samp_file)
    kde = get_kde(samp_df.q.values, samp_df.xeff.values)
    return kde


def calculate_p_with_kde(kde, params):
    print("Using KDE")
    for label, param in params.items():
        prob = kde([param['q'], param['xeff']])
        print(f"ln pi_qxeff({label} pt, {param})--> {np.log(prob/NUMERICAL_QXEFF_NORM_FACTOR)}")


def calculate_p_with_numerical(params):
    a1a2qcos2_prior = PriorDict(dict(
        a1=Uniform(0, 1),
        a2=Uniform(0, 1),
        q=Uniform(0, 1),
        cos2=Uniform(-1, 1)
    ))
    print("Using numerical")

    repetitions = 100
    odds = []
    odds_2 = []
    for i in range(repetitions):

        low_prob = p_param_and_xeff(
            param=LOW_Q['q'], xeff=LOW_Q['xeff'],
            init_a1a2qcos2_prior=a1a2qcos2_prior, param_key='q'
        )

        high_prob = p_param_and_xeff(
            param=HIGH_Q['q'], xeff=HIGH_Q['xeff'],
            init_a1a2qcos2_prior=a1a2qcos2_prior, param_key='q'
        )
        odd = np.log(high_prob/NUMERICAL_QXEFF_NORM_FACTOR) - np.log(low_prob/NUMERICAL_QXEFF_NORM_FACTOR)
        odds.append(odd)

        odds_2.append(np.log(high_prob/low_prob))

    print(f"ln pi odds {np.mean(odds):.2f} +/- {np.std(odds):.2f}")
    print(f"ln pi odds {np.mean(odds_2):.2f} +/- {np.std(odds_2):.2f}")


def main():
    params = get_max_l_data()
    calculate_p_with_kde(get_kde_object(SAMPLES_FILE), params)
    calculate_p_with_numerical(params)
    # print("ASTRO KDE")
    # calculate_p_with_kde(get_kde_object(ASTRO_SAMPLES_FILE), params)


if __name__ == '__main__':
    main()
