import glob
import json
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
    with open(path, 'r') as f:
        r = json.load(f)
    posterior = r['posterior']['content']
    data = posterior
    data.update({
        "log_evidence": r["log_evidence"],
        "log_evidence_err": r["log_evidence_err"],
        "q": posterior['mass_ratio'][0],
        "xeff": posterior['chi_eff'][0],
    })
    return data

def calc_numerical_posterior_odds(res1, res2):
    _, pri_o = _calc_prior_odds(res1, res2)
    post_o = posterior_odds(pri_o, res1["log_evidence"], res2["log_evidence"])
    return dict(prior_odds=pri_o, posterior_odds=post_o)

def get_results_and_compute_posterior_odds(res_regex):
    pri_odds, post_odds = {}, {}
    results = load_results(res_regex)
    if len(results)==0:
        print("No results found, cant compute posterior odds")
        return pri_odds, post_odds

    for rkey in list(combinations(results.keys(),r=2)):
        pt0, pt1 = results[rkey[0]], results[rkey[1]]
        _, pri_o = _calc_prior_odds(pt0, pt1)
        post_o = posterior_odds(pri_o, pt0["log_evidence"], pt1["log_evidence"])
        post_odds[f"{rkey[0]}:{rkey[1]}"] = post_o
        pri_odds[f"{rkey[0]}:{rkey[1]}"] = pri_o


    for k in post_odds.keys():
        print(f"Points {k}:")
        print(f">>> prior odds = {pri_odds[k]}")
        print(f">>> bayes fact = {post_odds[k]/pri_odds[k]}")
        print(f">>> postr odds = {post_odds[k]}")

    return pri_odds, post_odds



def posterior_odds(prior_odds_z0_by_z1, ln_z0, ln_z1):
    ln_bf = ln_z0 - ln_z1
    return prior_odds_z0_by_z1 * np.exp(ln_bf)
