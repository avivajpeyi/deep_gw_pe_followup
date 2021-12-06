import numpy as np
from mpi4py import MPI

from tqdm import tqdm
from ..prob_calculators import get_p_cos1_given_xeff_q_a1, get_p_a1_given_xeff_q


def mpi_p_cos1_given_a1_calc(cos1s, a1s, xeff, q, mcmc_n):
    data = dict(a1=np.array([]), cos1=np.array([]), p_cos1=np.array([]))
    for a1 in tqdm(a1s, desc=f"Building p_cos1 cache"):
        p_cos1_for_a1 = mpi_calc_p_cos1(cos1s, a1, xeff, q, mcmc_n)
        data['a1'] = np.append(data['a1'], np.array([a1 for _ in cos1s]))
        data['cos1'] = np.append(data['cos1'], cos1s)
        data['p_cos1'] = np.append(data['p_cos1'], p_cos1_for_a1)
    return data


def mpi_calc(func, x, *args):
    comm = MPI.COMM_WORLD
    pe = comm.Get_rank()  # identity of this process (process element, sometimes called rank)
    nprocs = comm.Get_size()  # number of processes
    root = nprocs - 1  # special process responsible for administrative work

    # total number of (work) elements
    n_global = len(x)

    # get the list of indices local to this process
    local_inds = np.array_split(np.arange(0, n_global), nprocs)[pe]

    # allocate and set local input values
    local_x = x[local_inds]

    local_res = [func(xi, *args) for xi in local_x]

    # gather all local arrays on process root, will
    # return a list of numpy arrays
    res_list = comm.gather(local_res, root=root)

    if pe == root:
        # turn the list of arrays into a single array
        res = np.concatenate(res_list)
        return res


def mpi_calc_p_cos1(cos1s, a1, xeff, q, mcmc_n):
    return mpi_calc(get_p_cos1_given_xeff_q_a1, cos1s, a1, xeff, q, mcmc_n)


def mpi_calc_p_a1(a1s, xeff, q, mcmc_n):
    return mpi_calc(get_p_a1_given_xeff_q, a1s, xeff, q, mcmc_n)
