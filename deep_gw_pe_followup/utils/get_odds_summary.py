from .calc_numerical_posterior_odds import calc_numerical_posterior_odds
from .calc_kde_odds import calc_kde_odds


def get_odds_summary(full, pt1, pt2, label):
    """Get the odds summary for a given point
    output: [label, kde_odds, priod_odds, bayes_factor, posterior_odds]
    """
    data = []
    data.append(label)
    kde_odds = calc_kde_odds(full.posterior, pt1, pt2)
    deep_odds = calc_numerical_posterior_odds(pt1, pt2)
    data.append(kde_odds)
    data.append(deep_odds['prior_odds'])
    data.append(deep_odds['bayes_factor'])
    data.append(deep_odds['posterior_odds'])
    return data
