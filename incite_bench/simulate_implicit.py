import numpy as np
import glob
from tqdm import tqdm
from mpi4py import MPI
import os
import sys
import json
from tqdm import tqdm
import antibody_design.utils.utils as utils
import argparse
import time
import numpy as np
from functools import wraps
from incite_bench.utils_bench import *
import antibody_design.simulate.simulate_implicit as simulate_implicit
import antibody_design.utils.fixpdb as fixpdb
#@mpi_timer
def simulate_bench(pdbs, out_general, rank, comm, time_file='time_sim_0.csv', num_nodes=1):
    _, rank, size = init_mpi()
    
    simulation_time = 50000

    #simulate_implicit.simulate_pipeline(
    #                        allpdbs_chunk,
    #                        f'{out_general}/{rank}',
    #                        fix_pdb=True,
    #                        simulation_time = simulation_time
    #                        )
    for it_s, pdb_it in enumerate(pdbs):
        os.makedirs(f'{out_general}/{it_s}/sims', exist_ok=True)
        if utils.is_folder_empty(f'{out_general}/{it_s}/sims'):
            try:
                fixpdb.pdbfixit(pdb_it, f'/dev/shm/{rank}_{it_s}.pdb')
                simulate_implicit.simulate_struct(
                                          f'/dev/shm/{rank}_{it_s}.pdb',
                                          simulation_time,
                                          f'{out_general}/{it_s}/sims/test.dcd',
                                          f'{out_general}/{it_s}/sims/test_out.pdb',
                                          f'{out_general}/{it_s}/sims/test.log',
                                          0)
            except Exception as e:
                print(e)
                print(pdb_it)
                continue
    return

def run(pdb_bench, out_rundir, tests_per_rank, num_nodes, time_fil_patt, use_mpi=True):
    if use_mpi==True:
        comm, rank_n, size = init_mpi()
        print(rank_n)
    else:
        comm = None
        rank_n = 0
    if tests_per_rank >1:
        pdbs = [pdb_bench for _ in range(tests_per_rank)]
    else:
        pdbs = [pdb_bench]
    out_rank = f'{out_rundir}/{rank_n}'
    os.makedirs(out_rank, exist_ok = True)
    time_file = f'{time_fil_patt}_{rank_n}.csv'
    simulate_bench(pdbs, out_rank, rank_n, comm, time_file=time_file, num_nodes=num_nodes)
    return comm, rank_n, size

def benchmark(args):
    # Open a file and read its contents as a string
    
    if args.no_mpi == True:
        use_mpi = False
    else:
        from mpi4py import MPI
        use_mpi = True

    # --- Start timing ---
    start_time = time.time()
    
    # Simulate some work (vary time per rank for demonstration)
    comm, rank_n, size = run(
                            args.inppdb,
                            args.out_run,
                            args.tests_per, 
                            args.num_nodes,
                            args.time_fil_patt,
                            use_mpi=use_mpi)
    
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
    parser.add_argument("-p", "--inppdb", type=str, help="File with sequence")
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


