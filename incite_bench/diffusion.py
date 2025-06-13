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
import incite_bench.binders_chroma as binders_chroma


def diff(
        input_pdbs: list[str],
        output_dir: str,  
        weights_backbone: str = '/dev/shm/chroma_weights/90e339502ae6b372797414167ce5a632/weights.pt',
        weights_design: str = '/dev/shm/chroma_weights/03a3a9af343ae74998768a2711c8b7ce/weights.pt',
        weights_conditioner: str = '/dev/shm/chroma_weights/3262b44702040b1dcfccd71ebbcf451d/weights.pt'):

    input_dir = './'
    hotspot = list(range(162, 173))
    """
    Divide receptors and get rank/size info using mpi
    Set ze_affinity_mask according to rank%ppn
    """

    reg_key = "794e2a0f3aca44bf9ad8108cd58894b9"
    device = "xpu"
    #pwd = "/lus/flare/projects/FoundEpidem/avasan/IDEAL/PeptideDesign/NMNAT_2_Screens"
    #input_dir = f"{pwd}/inputpdbs"
    #output_dir = f"{pwd}/Chroma/output_large/rank{rank}"
    
    len_binder = 100
    num_cycles = 4
    num_backbones = 10
    num_designs = 1
    diff_steps = 100
    
    chroma_bind_obj = binders_chroma.ChromaBinders(
                                    reg_key = reg_key,
                                    device = device,
                                    input_dir = input_dir,
                                    output_dir = output_dir,
                                    inp_pdbs = input_pdbs,
                                    len_binder = len_binder,
                                    num_cycles = num_cycles,
                                    num_backbones = num_backbones,
                                    num_designs = num_designs,
                                    diff_steps = diff_steps,
                                    hot_sphere = None,
                                    weights_backbone = weights_backbone,
                                    weights_design = weights_design,
                                    weights_conditioner = weights_conditioner,
                                    hotspot_indices = None,
                                    bind_sph_rad = None,
                                    centered_pdb_file = input_pdbs[0])
    chroma_bind_obj.run()
    return

def run(
        inp_pdb,
        out_rundir,
        tests_per_rank,
        num_nodes,
        time_fil_patt,
        use_mpi=True):

    if use_mpi==True:
        comm, rank_n, size = init_mpi()
        print(rank_n)
    else:
        comm = None
        rank_n = 0
    if tests_per_rank >1:
        pdbs = [inp_pdb for _ in range(tests_per_rank)]
    else:
        seqs = [inp_pdb]

    out_rank = f'{out_rundir}/{rank_n}'
    os.makedirs(out_rank, exist_ok = True)
    time_file = f'{time_fil_patt}_{rank_n}.csv'

    diff(pdbs,
        out_rank)

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
                            args.inp_pdb,
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
    parser.add_argument("-ip", "--inp_pdb", type=str, help="File with sequence")
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


