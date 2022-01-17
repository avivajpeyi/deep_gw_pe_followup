import os

DIR = os.path.dirname(__file__)


def get_mpl_style():
    return os.path.join(DIR, "plotting.mplstyle")
