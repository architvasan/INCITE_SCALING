import numpy as np
import os
import antibody_design.fold_ai.chai_pred as chai_pred
import pandas as pd
import json
from tqdm import tqdm
import antibody_design.utils.utils as utils
import argparse
import time
import numpy as np
from functools import wraps
from incite_bench.utils_bench import *

#@mpi_timer
def fold_bench(seqs, out_general, rank, comm, time_file='timer_0.csv', num_nodes=1):
    for it_s, s in enumerate(seqs):
        os.makedirs(f'{out_general}/{it_s}/chaiout', exist_ok=True)
        if utils.is_folder_empty(f'{out_general}/{it_s}/chaiout'):
            chai_pred.fold_chai([s],
                            ["protein"],
                            f'{out_general}/{it_s}/temp.fasta',
                            f'{out_general}/{it_s}/chaiout',
                            None,
                            devicetype='xpu')
    return

def run(seq_bench, out_rundir, tests_per_rank, num_nodes, time_fil_patt, use_mpi=True):
    if use_mpi==True:
        comm, rank_n, size = init_mpi()
        print(rank_n)
    else:
        comm = None
        rank_n = 0
    if tests_per_rank >1:
        seqs = [seq_bench for _ in range(tests_per_rank)]
    else:
        seqs = [seqs]
    out_rank = f'{out_rundir}/{rank_n}'
    os.makedirs(out_rank, exist_ok = True)
    time_file = f'{time_fil_patt}_{rank_n}.csv'
    fold_bench(seqs, out_rank, rank_n, comm, time_file=time_file, num_nodes=num_nodes)
    return comm, rank_n, size

def benchmark(args):
    # Open a file and read its contents as a string
    with open(args.seq_fil, "r") as file:
        seq = file.read()
    
    if args.no_mpi == True:
        use_mpi = False
    else:
        from mpi4py import MPI
        use_mpi = True

    # --- Start timing ---
    start_time = time.time()
    
    # Simulate some work (vary time per rank for demonstration)
    comm, rank_n, size = run(
                            seq,
                            args.out_run,
                            args.tests_per, 
                            args.num_nodes,
                            args.time_fil_patt,
                            use_mpi=use_mpi)
    
    # --- End timing ---
    end_time = time.time()

    duration = end_time - start_time
    print(f"rank: {rank_n}, duration: {duration}") 
    # Prepare the CSV line as bytes
    output_line = f"{rank_n},{duration:.6f}\n"
    output_bytes = output_line.encode('utf-8')
    line_size = len(output_bytes)
    
    # Ensure all lines are the same size for safe offset calculation
    max_line_size = comm.allreduce(line_size, op=MPI.MAX)
    output_bytes = output_bytes.ljust(max_line_size, b' ')
    
    # Each rank calculates its offset
    offset = rank_n * max_line_size
    
    # Parallel write to a shared CSV file
    file = MPI.File.Open(comm, f"{args.time_fil_patt}_{args.num_nodes}.csv",
                         MPI.MODE_CREATE | MPI.MODE_WRONLY)
    file.Write_at(offset, output_bytes)
    file.Close()
    
    # Optionally, have rank 0 write a header (separate serial step)
    if rank_n == 0:
        with open(f"{args.time_fil_patt}_{args.num_nodes}.csv", "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write("rank,duration_seconds\n" + content)
    
    if rank_n == 0:
        print("Timing results written to 'timing_results.csv'")

def main():
    parser = argparse.ArgumentParser(description="A simple argparse example")

    # Add arguments
    parser.add_argument("-s", "--seq_fil", type=str, help="File with sequence")
    parser.add_argument("-o", "--out_run", type=str, help="Directory to output all data")
    parser.add_argument("-t", "--tests_per", type=int, help="Num tests per rank")
    parser.add_argument("-N", "--num_nodes", type=int, help="Number of nodes")
    parser.add_argument("-tf", "--time_fil_patt", type=str, help="Time file pattern")
    parser.add_argument("--no_mpi", action="store_true", help="use_mpi")

    # Parse arguments
    args = parser.parse_args()

    benchmark(args)
if __name__ == "__main__":
    main()
