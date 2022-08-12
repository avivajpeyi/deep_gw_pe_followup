import sys
import os

from parallel_bilby.generation import main as pbilby_main
from parallel_bilby.generation import (create_generation_parser, get_cli_args)
from deep_gw_pe_followup.utils.adjust_pbilby_prior_json import adjust_prior

def main():
    # run parallel_bilby_generation ini
    pbilby_main()

    # fix up prior json
    cli_args = get_cli_args()
    generation_parser = create_generation_parser()
    args = generation_parser.parse_args(args=cli_args)
    prior_json_fname = f"{args.outdir}/data/{args.label}_prior.json"
    adjust_prior(prior_json_fname)
    print("READY FOR DEEP-FOLLOWUP")
