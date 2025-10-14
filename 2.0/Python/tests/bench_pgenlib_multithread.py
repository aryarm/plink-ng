#!/usr/bin/env python3
import time
import concurrent.futures
import numpy as np
import pgenlib
import pathlib
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

def main():
    nvariant_limits = np.linspace(7000, 12000, 10, dtype=int)
    single_times = []
    multi_times = []
    for nvariant_limit in nvariant_limits:
        pgen_dir = f"temp/sample_{nvariant_limit}"
        pgen_path = generate_large_pgen(pgen_dir, nvariant_min=1, nvariant_limit=nvariant_limit)
        single, single_arr = timed_read(pgen_path)
        multi, multi_arr = threaded_timed_read(pgen_path)
        np.testing.assert_allclose(single_arr, multi_arr)
        single_times.append(single)
        multi_times.append(multi)
        print(f"nvariant_limit={nvariant_limit}: single={single:.2f}s, multi={multi:.2f}s")

    # Fit OLS lines
    X = nvariant_limits.reshape(-1, 1)
    single_model = LinearRegression().fit(X, single_times)
    multi_model = LinearRegression().fit(X, multi_times)
    single_slope = single_model.coef_[0]
    multi_slope = multi_model.coef_[0]

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(nvariant_limits, single_times, 'o-', label='Single-threaded')
    plt.plot(nvariant_limits, multi_times, 'o-', label='Multi-threaded')
    plt.plot(nvariant_limits, single_model.predict(X), '--', label=f'Single OLS (slope={single_slope:.4f})')
    plt.plot(nvariant_limits, multi_model.predict(X), '--', label=f'Multi OLS (slope={multi_slope:.4f})')
    plt.xlabel('nvariant_limit')
    plt.ylabel('Read time (s)')
    plt.title('Single vs Multi-threaded Read Timings')
    plt.legend()
    plt.tight_layout()
    plt.savefig('temp/timings_plot.png')
    print(f"Single-threaded OLS slope: {single_slope:.6f} s/variant")
    print(f"Multi-threaded OLS slope:  {multi_slope:.6f} s/variant")
    print("Plot saved as timings_plot.png")

if __name__ == "__main__":
    main()
