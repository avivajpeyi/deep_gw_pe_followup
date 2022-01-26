"""benchmark the drawing of samples from prior"""
import time
t0 = time.time()
from bilby.gw.prior import BBHPriorDict, PriorDict
print(f"Time to import bilby: {time.time()-t0}s")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N = np.logspace(2, 7)


def time_sample_draws(i, p: PriorDict):
    t0 = time.time()
    p.sample(int(i))
    return time.time() - t0


def benchmark(prior, n=N):
    times = []
    processed_n = []
    for i, n_ in tqdm(enumerate(n), total=len(N)):
        runtime = time_sample_draws(n_, prior)
        times.append(runtime)
        processed_n.append(n_)
        if i > 3:
            plt.close('all')
            plt.scatter(times, processed_n)
            plt.xlabel("time (s)")
            plt.ylabel("N prior samples")
            plt.xscale('log')
            plt.tight_layout()
            plt.savefig(f"benchmark.png")


if __name__ == '__main__':
    benchmark(BBHPriorDict(), N)