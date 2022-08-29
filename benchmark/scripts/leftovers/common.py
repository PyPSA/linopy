import tracemalloc
from subprocess import PIPE
from time import sleep, time

import pandas as pd
import psutil as ps
from memory_profiler import memory_usage


def profile(nrange, func):
    res = pd.DataFrame(index=nrange, columns=["Time [s]", "Memory Usage"])

    for N in res.index:
        start = time()

        func(N)

        end = time()
        duration = end - start

        memory = memory_usage((func, (N,)))

        res.loc[N] = duration, max(memory)

    return res


def profile_shell(nrange, cmd):
    res = pd.DataFrame(index=nrange, columns=["Time [s]", "Memory Usage"])

    for N in res.index:
        tracemalloc.start()
        start = time()

        process = ps.Popen(cmd(N).split(), stdout=PIPE)

        peak_mem = 0
        peak_cpu = 0

        # while the process is running calculate resource utilization.
        while process.is_running():
            # set the sleep time to monitor at an interval of every second.
            sleep(0.01)

            # capture the memory and cpu utilization at an instance
            mem = process.memory_info().rss
            cpu = process.cpu_percent()

            # track the peak utilization of the process
            if mem > peak_mem:
                peak_mem = mem
            if cpu > peak_cpu:
                peak_cpu = cpu
            if mem == 0.0:
                break

        end = time()
        duration = end - start

        res.loc[N] = duration, peak_mem

    return res
