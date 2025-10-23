#!/usr/bin/env python3
import os
import time
import pgenlib
import pathlib
import numpy as np
import concurrent.futures
import multiprocessing as mpi
import matplotlib.pyplot as plt
from test_pgenlib import unphased_biallelic_case
from sklearn.linear_model import LinearRegression

num_cpus = len(os.sched_getaffinity(os.getpid()))
print(f"Using {num_cpus} CPUs")

def generate_large_pgen(pgen_dir, case_idx=0, nsample_min=1, nsample_limit=20000, nvariant_min=1, nvariant_limit=80000):
    pgen_dir = pathlib.Path(pgen_dir)
    if pgen_dir.exists():
        return next(pgen_dir.glob("*.pgen"))  # Return existing .pgen file if present
    pgen_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    # Generate a single large pgen file
    start = time.time()
    unphased_biallelic_case(pgen_dir, case_idx, nsample_min, nsample_limit, nvariant_min, nvariant_limit)
    print(f"Time to generate PGEN: {time.time() - start:.2f}s")
    # Return the expected .pgen file path
    return next(pgen_dir.glob("*.pgen"))

def timed_read(file_path, return_arr=True):
    with pgenlib.PgenReader(str(file_path).encode()) as r:
        num_vars = r.get_variant_ct()
        arr = np.empty([num_vars, r.get_raw_sample_ct()*2], dtype=np.int32)
        start = time.time()
        r.read_alleles_range(0, num_vars, arr)
        elapsed = time.time() - start
    if return_arr:
        return elapsed, arr
    else:
        return elapsed

def threaded_timed_read(file_path, n_threads=num_cpus, single_arr=None):
    file_path = str(file_path).encode()
    with pgenlib.PgenReader(file_path) as r:
        num_vars, n_samples = r.get_variant_ct(), r.get_raw_sample_ct()
        arr = np.empty((num_vars, n_samples * 2), dtype=np.int32)

    def do_reads(s, e):
        with pgenlib.PgenReader(file_path) as o:
            o.read_alleles_range(s, e, arr[s:e])

    start = time.time()
    chunk = (num_vars + n_threads - 1) // n_threads
    with concurrent.futures.ThreadPoolExecutor(n_threads) as ex:
        ex.map(lambda i: do_reads(i * chunk, min((i + 1) * chunk, num_vars)), range(n_threads))
    elapsed = time.time() - start
    if single_arr is not None:
        np.testing.assert_allclose(single_arr, arr)
    return elapsed

def process_init_mp(
    file_path_: str = None,
    num_vars_: int = None,
    n_samples_: int = None,
    shared_arr_: mpi.Array = None,
):
    """
    A helper method that globalizes certain variables so that they can be used in
    multiprocessing. This method should only be called in each parallel
    child/worker process and not the parent process to avoid polluting the global
    namespace.

    This is preferable to passing these values as arguments to the _do_reads method
    because, otherwise, python will try to pickle the arguments

    Parameters
    ----------
    shared_arr: mpi.Array
        The underlying bytes of the matrix, as a shared-memory Array
    """
    global shared_arr, num_vars, n_samples, file_path
    shared_arr, num_vars, n_samples, file_path = shared_arr_, num_vars_, n_samples_, file_path_

def process_do_reads(s, e):
    shd_arr = shared_arr.get_obj()
    with pgenlib.PgenReader(file_path) as o:
        arr = np.frombuffer(shd_arr, dtype=np.int32).reshape((num_vars, n_samples * 2))
        o.read_alleles_range(s, e, arr[s:e])

def process_timed_read(file_path, n_threads=num_cpus, single_arr=None):
    return 0
    file_path = str(file_path).encode()
    with pgenlib.PgenReader(file_path) as r:
        num_vars, n_samples = r.get_variant_ct(), r.get_raw_sample_ct()
        shared_arr = mpi.Array("i", int(np.prod((num_vars, n_samples * 2))))
        shd_arr = shared_arr.get_obj()
        arr = np.frombuffer(shd_arr, dtype=np.int32).reshape((num_vars, n_samples * 2))

    start = time.time()
    chunk = (num_vars + n_threads - 1) // n_threads
    pairs = [(i * chunk, min((i + 1) * chunk, num_vars)) for i in range(n_threads)]
    mp_chunksize = int(np.ceil(len(pairs) / n_threads))
    with mpi.Pool(
        processes=n_threads,
        initargs=(file_path, num_vars, n_samples, shared_arr),
        initializer=process_init_mp,
    ) as ex:
        ex.starmap(process_do_reads, pairs, chunksize=mp_chunksize)
    elapsed = time.time() - start
    if single_arr is not None:
        np.testing.assert_allclose(single_arr, arr)
    return elapsed

def bench_instance(pgen_path: pathlib.Path):
    # single, single_arr = timed_read(pgen_path)
    # thread = threaded_timed_read(pgen_path, single_arr=single_arr)
    # proc = process_timed_read(pgen_path, single_arr=single_arr)

    single = timed_read(pgen_path, return_arr=False)
    thread = threaded_timed_read(pgen_path)
    proc = process_timed_read(pgen_path)

    return single, thread, proc

def main():
    nsample_fixed = 200000
    nvariant_limits = np.linspace(5, 118, 10, dtype=int)
    single_times = []
    thread_times = []
    process_times = []

    for nvariant_limit in nvariant_limits:
        pgen_dir = f"temp/sample_{nvariant_limit}"
        pgen_path = generate_large_pgen(pgen_dir, nvariant_limit=nvariant_limit, nsample_limit=nsample_fixed)

        single_times_rep = []
        thread_times_rep = []
        process_times_rep = []

        for rep in range(5):
            single_times_rep, thread_times_rep, process_times_rep = bench_instance(pgen_path)

        single_times.append(np.mean(single_times_rep))
        thread_times.append(np.mean(thread_times_rep))
        process_times.append(np.mean(process_times_rep))

        print(
            f"nvariant_limit={nvariant_limit}: "
            f"single={single_times_rep:.2f}s, "
            f"thread={thread_times_rep:.2f}s, "
            f"process={process_times_rep:.2f}s"
        )

    # Fit OLS models
    X = nvariant_limits.reshape(-1, 1)
    models = {
        "Single-threaded": (single_times, LinearRegression().fit(X, single_times)),
        "Multi-threaded": (thread_times, LinearRegression().fit(X, thread_times)),
        "Multi-process": (process_times, LinearRegression().fit(X, process_times)),
    }

    plt.figure(figsize=(8,6))
    for label, (times, model) in models.items():
        slope = model.coef_[0]
        if slope == 0:
            continue
        print(f"{label} slope: {slope}")
        line, = plt.plot(nvariant_limits, times, 'o-', label=f"{label} (slope={slope:.5e})")
        plt.plot(nvariant_limits, model.predict(X), '--', color=line.get_color())
    plt.xlabel(f"Number of variants\nNumber of samples fixed at {nsample_fixed}")
    plt.ylabel('Read time (s)')
    plt.title('PgenReader Timings')
    plt.legend()
    plt.tight_layout()
    plt.savefig('temp/timings_plot.png')
    print("Plot saved as timings_plot.png")

if __name__ == "__main__":
    main()
