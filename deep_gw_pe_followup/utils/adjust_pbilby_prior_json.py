from deep_gw_pe_followup.restricted_prior.prior import RestrictedPrior
import os
import sys
from bilby.core.prior import PriorDict

def adjust_prior(json_path):
    """{self.data_directory}/{self.label}_prior.json"""
    prior = PriorDict.from_json(json_path)
    prior = RestrictedPrior.from_bbh_priordict(prior)
    prior.plot_cache()
    outdir = os.path.dirname(json_path)
    label = os.path.basename(json_path).split("_prior.json")[0]
    prior.to_json(outdir=outdir, label=label)
    print("Re-adjusted prior for deep-followup")

def main():
    filename = sys.argv[-1]
    adjust_prior(filename)


if __name__ == '__main__':
    main()
