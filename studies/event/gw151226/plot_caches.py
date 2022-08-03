from deep_gw_pe_followup.restricted_prior import RestrictedPrior

if __name__ == "__main__":
    highq_prior = RestrictedPrior(filename="priors/lvk_restricted.prior")
    highq_prior.plot_cache()
    ias_prior = RestrictedPrior(filename="priors/ias_restricted.prior")
    ias_prior.plot_cache()
