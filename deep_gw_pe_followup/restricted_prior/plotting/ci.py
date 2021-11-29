import numpy as np
import pandas as pd


def plot_ci(x, y, ax, label='', alpha=0.7, zorder=-10):
    cols = ['#EE7550', '#F19463', '#F6B176']
    ci = bin_by(x, y)
    # plot the 3rd stdv
    ax.fill_between(ci.x, ci['5th'], ci['95th'], alpha=alpha, color=cols[2], zorder=zorder - 3)
    ax.fill_between(ci.x, ci['10th'], ci['90th'], alpha=alpha, color=cols[1], zorder=zorder - 2)
    ax.fill_between(ci.x, ci['25th'], ci['75th'], alpha=alpha, color=cols[0], zorder=zorder - 1)
    # plt the line
    ax.plot(ci.x, ci['median'], color=cols[0], label=label, zorder=zorder)


def bin_by(x, y, nbins=30, bins=None):
    """
    Divide the x axis into sections and return groups of y based on its x value
    """
    if bins is None:
        bins = np.linspace(x.min(), x.max(), nbins)

    bin_space = (bins[-1] - bins[0]) / (len(bins) - 1) / 2

    indicies = np.digitize(x, bins + bin_space)

    output = []
    for i in range(0, len(bins)):
        output.append(y[indicies == i])
    #
    # prepare a dataframe with cols: median; mean; 1up, 1dn, 2up, 2dn, 3up, 3dn
    df_names = ['mean', 'median', '5th', '95th', '10th', '90th', '25th', '75th']
    df = pd.DataFrame(columns=df_names)
    to_delete = []
    # for each bin, determine the std ranges
    for y_set in output:
        if y_set.size > 0:
            av = y_set.mean()
            intervals = np.percentile(y_set, q=[50, 5, 95, 10, 90, 25, 75])
            res = [av] + list(intervals)
            df = df.append(pd.DataFrame([res], columns=df_names))
        else:
            # just in case there are no elements in the bin
            to_delete.append(len(df) + 1 + len(to_delete))

    # add x values
    bins = np.delete(bins, to_delete)
    df['x'] = bins

    return df
