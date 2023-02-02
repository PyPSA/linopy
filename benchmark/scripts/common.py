import gc
from time import time

import pandas as pd

# from memory_profiler import memory_usage


def profile(nrange, func, *args):
    res = pd.DataFrame(index=nrange, columns=["Time", "Memory", "Objective"])

    for N in res.index:
        start = time()

        objective = func(N, *args)

        end = time()
        duration = end - start

        res.loc[N, "Time"] = duration
        res.loc[N, "Objective"] = objective

        gc.collect()

    return res
