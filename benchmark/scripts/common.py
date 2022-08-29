import gc
from time import time

import pandas as pd
from memory_profiler import memory_usage


def profile(nrange, func, *args):
    res = pd.DataFrame(index=nrange, columns=["Time", "Memory"])

    for N in res.index:
        start = time()

        func(N, *args)

        end = time()
        duration = end - start

        memory = memory_usage((func, (N, *args)))

        res.loc[N] = duration, max(memory)

        gc.collect()

    return res
