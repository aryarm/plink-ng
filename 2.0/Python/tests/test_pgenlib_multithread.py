#!/usr/bin/env python3
import time
import concurrent.futures
import numpy as np
import pgenlib
import pathlib
from test_pgenlib import phased_multiallelic_case

def generate_large_pgen(pgen_dir, case_idx=0, nsample_min=1, nsample_limit=6000, nvariant_min=1, nvariant_limit=10000, allele_ct_max=2):
    pgen_dir = pathlib.Path(pgen_dir)
    pgen_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    # Generate a single large pgen file
    start = time.time()
    phased_multiallelic_case(pgen_dir, case_idx, nsample_min, nsample_limit, nvariant_min, nvariant_limit, allele_ct_max)
    elapsed = time.time() - start
    print(f"Generated .pgen file: {elapsed:.2f} seconds")
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
    with pgenlib.PgenReader(str(file_path).encode()) as r:
        num_vars = r.get_variant_ct()
        arr = np.empty([num_vars, r.get_raw_sample_ct()*2], dtype=np.int32)
        def do_reads(start, end):
            with pgenlib.PgenReader(str(file_path).encode()) as o:
                o.read_alleles_range(start, end, arr[start:end])
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            chunk_size = (num_vars + n_threads - 1) // n_threads
            for i in range(n_threads):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, num_vars)
                futures.append(executor.submit(do_reads, start_idx, end_idx))
            for future in futures:
                future.result()
        elapsed = time.time() - start
    return elapsed, arr

 # Generate or reuse a large .pgen file first
pgen_path = generate_large_pgen("sample")

single, single_arr = timed_read(pgen_path)
print(f"Single-threaded read: {single:.2f} seconds")

multi, multi_arr = threaded_timed_read(pgen_path)
print(f"Multi-threaded read:  {multi:.2f} seconds")

np.testing.assert_allclose(single_arr, multi_arr)
