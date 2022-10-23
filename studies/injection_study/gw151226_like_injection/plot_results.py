
import json
import pandas as pd
from bilby.gw.conversion import generate_all_bbh_parameters
import matplotlib.pyplot as plt
from corner.core import hist2d

from deep_gw_pe_followup.utils.calc_numerical_posterior_odds import get_results_and_compute_posterior_odds
from deep_gw_pe_followup.utils.calc_kde_odds import calc_kde_odds

RES = dict(
    regular="general_pe/out_nlive2000_nact10/result/GW151226_injection_0_result.json",
    pt1="followup_pt1/out_followup1/result/followup1_0_result.json",
    pt2="followup_pt2/out_followup2/result/followup2_0_result.json"
)

PT1 = dict(mass_ratio=0.3605278878376452, chi_eff=0.1729095104595008)
PT2 = dict(mass_ratio=0.3923, chi_eff=0.2461)


TRUE = {'luminosity_distance': 800, 'mass_ratio': 0.26512986754630535, 'chirp_mass': 9.702165419247658,
        'a_1': 0.9873737113225366, 'a_2': 0.8435528737659149, 'tilt_1': 1.0401383271620692,
        'tilt_2': 2.5427989903091914, 'phi_12': 1.695426569783911, 'phi_jl': 3.3438963165263136,
        'dec': 0.33364118287594846, 'ra': 0.6738167456098734, 'cos_theta_jn': -0.8684204854784665,
        'psi': 2.06954952383973, 'phase': 0.8866172464617089, 'geocent_time': 0, 'theta_jn': 2.6228040981756555}
TRUE = generate_all_bbh_parameters(TRUE)


def from_json(filename, param):
    with open(filename, "r") as file:
        dictionary = json.load(file)
    dictionary = {k: dictionary[k] for k in ["log_evidence", "log_bayes_factor", "posterior"]}
    dictionary['posterior'] = pd.DataFrame(dictionary['posterior']['content'])[param]
    return dictionary


def read_res(param):
    return {k: from_json(v, param) for k, v in RES.items()}


def make_corner():
    from agn_utils.plotting import overlaid_corner
    param = ["chi_eff", "mass_ratio", "mass_1", "mass_2", "cos_tilt_1", "cos_tilt_2"]
    ranges = [(0.1, 0.4), (0.2, 0.6), (15, 25), (5, 9), (0, 1), (-1, 1)]
    res = read_res(param)
    print("Results read in")
    overlaid_corner(
        samples_list=[p['posterior'] for p in res.values()],
        params=param,
        samples_colors=["C0", "C1", "C2"],
        sample_labels=[f"{k} (lnZ {r['log_bayes_factor']:.2f})" for k, r in res.items()],
        fname="compare.png",
        ranges=ranges,
        truths=TRUE,
        override_kwargs=dict(truth_color="k")
    )
    print("plot saved")

    for i, (k, r) in enumerate(res.items()):
        plt.close('all')
        plt.hist(r['posterior']['mass_2'], histtype='step', color=f"C{i}")
        plt.xlabel("mass 2")
        plt.tick_params(labelleft=False, left=False)
        plt.tight_layout()
        plt.savefig(f"{k}_mass2.png")



def main():
    # get_results_and_compute_posterior_odds("followup_pt*/out*/res*/*.json")
    full_res = dict(
        q=from_json(RES['regular'], "mass_ratio")['posterior'],
        xeff=from_json(RES['regular'], "chi_eff")['posterior'],
    )
    # kde_odds = calc_kde_odds(full_res, PT1, PT2)
    # print("kde_odds = ", kde_odds)
    fig, ax = plt.subplots(1,1, figsize=(4,4))
    hist2d(full_res['q'].values, full_res['xeff'].values, color="tab:orange", data_kwargs=dict(alpha=0.05), plot_density=False, fill_contours=True, smooth=0.8)
    plt.plot(PT1['mass_ratio'], PT1['chi_eff'], marker='$A$', color='b',  zorder=10)
    plt.plot(PT2['mass_ratio'], PT2['chi_eff'], marker='$B$', color='r',  zorder=10)
    plt.xlabel(r"$q$")
    plt.ylabel(r"$\chi_{\rm eff}$")
    plt.text(0.5, 0.7, "A VS B\nkde = 0.5\ndeep = 0.8", transform=ax.transAxes)
    plt.savefig("kde_odds.png")


if __name__ == '__main__':
    main()