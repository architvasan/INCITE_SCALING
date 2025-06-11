#!/usr/bin/env python
from glob import glob
from math import ceil
import os
from mpi4py import MPI
from incite_bench.utils_bench import *
import time

def run_md(path: str, steps: int, eq_steps: int =500000, prod_log = None):
    from molecular_simulations.simulate.omm_simulator import Simulator

    simulator = Simulator(path, prod_log = prod_log, prod_steps=steps, platform='OpenCL')
    simulator.run()

def run(syst_bench, num_ns, num_nodes, prodlog_dir, time_fil_patt, use_mpi=True):
    import shutil
    """
    initialize mpi
    """
    if use_mpi==True:
        comm, rank_n, size = init_mpi()
        print(rank_n)
    else:
        comm = None
        rank_n = 0
    """
    copy system files to /dev/shm
    """
    dest_dir = f"/dev/shm/syst_{rank_n}"
    shutil.copytree(syst_bench, dest_dir, dirs_exist_ok=True)
    
    if not os.path.isdir(prodlog_dir):
        os.makedirs(prodlog_dir, exist_ok = True)

    """
    run test sims
    """
    #time_file = f'{time_fil_patt}_{rank_n}.csv'
    timestep = 4
    n_steps = num_ns / timestep * 1000000
    run_md(dest_dir, steps = n_steps, prod_log = f'{prodlog_dir}/prod_rnk{rank_n}.log')
    return comm, rank_n, size

def benchmark(args):

    # --- Start timing ---
    start_time = time.time()

    # timeout variable can be omitted, if you use specific value in the while condition
    timeout = 300   # [seconds]
    
    timeout_start = time.time()
    
    while time.time() < timeout_start + timeout:
        # Simulate some work (vary time per rank for demonstration)
        comm, rank_n, size = run(
                              args.syst,
                              args.num_ns,
                              args.num_nodes,
                              args.prodlog_dir,
                              args.time_fil_patt)
        
        # --- End timing ---
        end_time = time.time()
    
        duration = end_time - start_time
        
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
    parser.add_argument("-s", "--syst", type=str, help="System files location")

    parser.add_argument("-n", "--num_ns", type=int, help="Number of ns to simulate for")

    parser.add_argument("-N", "--num_nodes", type=int, help="Number of nodes")

    parser.add_argument("-pd", "--prodlog_dir", type=str, help="Directory to record production log files")

    parser.add_argument("-tf", "--time_fil_patt", type=str, help="Time file pattern")

    parser.add_argument("--no_mpi", action="store_true", help="use_mpi")

    # Parse arguments
    args = parser.parse_args()

    benchmark(args)


if __name__ == "__main__":
    main()








