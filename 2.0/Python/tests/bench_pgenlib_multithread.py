#!/usr/bin/env python3
import time
import pgenlib
import pathlib
import numpy as np
import concurrent.futures
import multiprocessing as mpi
import matplotlib.pyplot as plt
from test_pgenlib import phased_multiallelic_case
from sklearn.linear_model import LinearRegression

def generate_large_pgen(pgen_dir, case_idx=0, nsample_min=1, nsample_limit=6000, nvariant_min=1, nvariant_limit=10000, allele_ct_max=2):
    pgen_dir = pathlib.Path(pgen_dir)
    if pgen_dir.exists():
        return next(pgen_dir.glob("*.pgen"))  # Return existing .pgen file if present
    pgen_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    # Generate a single large pgen file
    start = time.time()
    phased_multiallelic_case(pgen_dir, case_idx, nsample_min, nsample_limit, nvariant_min, nvariant_limit, allele_ct_max)
    elapsed = time.time() - start
    # Return the expected .pgen file path
    return next(pgen_dir.glob("*.pgen"))

def timed_read(file_path):
    with pgenlib.PgenReader(str(file_path).encode()) as r:
        num_vars = r.get_variant_ct()
        arr = np.empty([num_vars, r.get_raw_sample_ct()*2], dtype=np.int32)
        start = time.time()
        r.read_alleles_range(0, num_vars, arr)
        elapsed = time.time() - start
    return elapsed, arr

def threaded_timed_read(file_path, n_threads=4):
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
    return time.time() - start, arr

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

def process_timed_read(file_path, n_threads=4):
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
    return time.time() - start, arr

def main():
    nvariant_limits = np.linspace(7000, 12000, 10, dtype=int)
    single_times = []
    multi_times = []
    process_times = []

    for nvariant_limit in nvariant_limits:
        pgen_dir = f"temp/sample_{nvariant_limit}"
        pgen_path = generate_large_pgen(pgen_dir, nvariant_min=1, nvariant_limit=nvariant_limit)
        single, single_arr = timed_read(pgen_path)
        multi, multi_arr = threaded_timed_read(pgen_path)
        proc, proc_arr = process_timed_read(pgen_path)

        np.testing.assert_allclose(single_arr, multi_arr)
        np.testing.assert_allclose(single_arr, proc_arr)

        single_times.append(single)
        multi_times.append(multi)
        process_times.append(proc)

        print(f"nvariant_limit={nvariant_limit}: single={single:.2f}s, "
              f"multi={multi:.2f}s, process={proc:.2f}s")

    # Fit OLS models
    X = nvariant_limits.reshape(-1, 1)
    models = {
        "Single-threaded": (single_times, LinearRegression().fit(X, single_times)),
        "Multi-threaded": (multi_times, LinearRegression().fit(X, multi_times)),
        "Multi-process": (process_times, LinearRegression().fit(X, process_times)),
    }

    plt.figure(figsize=(8,6))
    for label, (times, model) in models.items():
        slope = model.coef_[0]
        print(f"{label} slope: {slope}")
        plt.plot(nvariant_limits, times, 'o-', label=f"{label} (slope={slope:.5e})")
        plt.plot(nvariant_limits, model.predict(X), '--')
    plt.xlabel('nvariant_limit')
    plt.ylabel('Read time (s)')
    plt.title('Single vs Multi-threaded vs Multi-process Read Timings')
    plt.legend()
    plt.tight_layout()
    plt.savefig('temp/timings_plot.png')
    print("Plot saved as timings_plot.png")

if __name__ == "__main__":
    main()
