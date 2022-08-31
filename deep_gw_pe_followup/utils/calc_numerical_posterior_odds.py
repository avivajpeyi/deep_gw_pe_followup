import glob
from bilby.gw.result import CBCResult
import os
import numpy as np
from typing import Dict
from itertools import combinations
from .isotropic_spins_prior_odds import _calc_prior_odds



def load_results(res_regex)->Dict:
    """{ptA: CBCResult, ptB: CBCResult...}"""
    result_files = glob.glob(res_regex)
    names = [os.path.basename(p).split("_")[0] for p in result_files]
    loaded_results = {}
    for  p, n in zip(result_files, names):
        try:
            loaded_results[n] = extract_res_info(p)
        except Exception as e:
            pass
    return loaded_results

def extract_res_info(path):
    r = CBCResult.from_json(path)
    return {
        "log_evidence": r.log_evidence,
        "log_evidence_err": r.log_evidence_err,
        "q": r.priors['mass_ratio'].peak,
        "xeff": r.priors['chi_eff'].peak,
    }


def get_results_and_compute_posterior_odds(res_regex):
    odds = {}
    results = load_results(res_regex)
    if len(results)==0:
        print("No results found, cant compute posterior odds")
        return odds

    print("Posterior odds: ")
    for rkey in list(combinations(results.keys(),r=2)):
        pt0, pt1 = results[rkey[0]], results[rkey[1]]
        _, pri_o = _calc_prior_odds(pt0, pt1)
        post_o = posterior_odds(pri_o, pt0.log_evidence, pt1.log_evidence)
        odds[f"{rkey[0]}:{rkey[1]}"] = post_o
        print(f">>> O({rkey[0]}/{rkey[1]} = {post_o}")
    return odds



def posterior_odds(prior_odds_z0_by_z1, ln_z0, ln_z1):
    ln_bf = ln_z0 - ln_z1
    return prior_odds_z0_by_z1 * np.exp(ln_bf)
