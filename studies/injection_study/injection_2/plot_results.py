from agn_utils.plotting import overlaid_corner
import json
import pandas as pd
from bilby.gw.conversion import generate_all_bbh_parameters
import matplotlib.pyplot as plt

RES = dict(
    regular="general_pe/outdir/result/GW151226_injection_0_result.json",
    pt1="followup_pt1/out_followup1/result/followup1_0_result.json",
    pt2="followup_pt2/out_followup2/result/followup2_0_result.json"
)

TRUE = {'luminosity_distance': 800, 'mass_ratio': 0.26512986754630535, 'chirp_mass': 9.702165419247658, 'a_1': 0.9873737113225366, 'a_2': 0.8435528737659149, 'tilt_1': 1.0401383271620692, 'tilt_2': 2.5427989903091914, 'phi_12': 1.695426569783911, 'phi_jl': 3.3438963165263136, 'dec': 0.33364118287594846, 'ra': 0.6738167456098734, 'cos_theta_jn': -0.8684204854784665, 'psi': 2.06954952383973, 'phase': 0.8866172464617089, 'geocent_time': 0, 'theta_jn': 2.6228040981756555}
TRUE = generate_all_bbh_parameters(TRUE)

def from_json(filename, param):
    with open(filename, "r") as file:
        dictionary = json.load(file)
    dictionary = {k: dictionary[k] for k in ["log_evidence", "log_bayes_factor", "posterior"]}
    dictionary['posterior'] = pd.DataFrame(dictionary['posterior']['content'])[param]
    return dictionary


def read_res(param):
    return {k: from_json(v, param) for k, v in RES.items()}


if __name__ == '__main__':
    param = ["chi_eff", "mass_ratio", "mass_1", "mass_2", "cos_tilt_1", "cos_tilt_2"]
    ranges = [(0.1,0.4), (0.2,0.6), (15,25), (5,9), (0,1), (-1,1)]
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

