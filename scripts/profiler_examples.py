# external imports
import sys, os, gc
import numpy as np

pysrc = os.path.join(os.path.dirname(__file__), "..", "pysrc")
sys.path.append(pysrc)
from cudart import *

if __name__ == "__main__":

    # example usage for invoking profiler and viewing results

    data_dir = "/mnt/kocsis1/cuDART_wdir/profiling/data"
    prof_dir = "/mnt/kocsis1/cuDART_wdir/profiling/profiles_16"
    timing_str = "/mnt/kocsis1/cuDART_wdir/profiling/timings.npy"
    timing_png = "/mnt/kocsis1/cuDART_wdir/profiling/timings.png"
    D_span = [64,128,256,512]
    N_span = [64,128,256,512,1024,2048]
    num_iter = 10
    profiler = Profiler(data_dir = data_dir, prof_dir = prof_dir)
    #profiler.build(D_span = D_span, save_boosted = True)
    profiler.run(N_span = N_span, D_span = D_span, num_iter=num_iter)
    profiler.save_timings(save_str = timing_str, N_span = N_span, D_span = D_span)
    # timing_str = "/scratch/github/cuDART_wdir/profiling/timings.npy"
    # timing_png = "/scratch/github/cuDART_wdir/profiling/timings.png"
    profiler.plot_timings(load_str = timing_str, save_str = timing_png, N_span = N_span, D_span = D_span, num_iter=num_iter)