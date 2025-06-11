#!/usr/bin/env python
from glob import glob
from math import ceil
import os
from mpi4py import MPI

def init_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Get the rank of the current process
    size = comm.Get_size() # Get the total number of processes
    return comm, rank, size

def run_md(path: str, eq_steps=500_000, steps=250_000_000):
    from molecular_simulations.simulate.omm_simulator import Simulator

    simulator = Simulator(path, equil_steps=eq_steps, prod_steps=steps, platform='OpenCL')
    simulator.run()


if __name__ == "__main__":
    comm, rank, size = init_mpi()
    throw_chai = [(0, 0), (4, 4)]
    throw_boltz = [(0, 1), (2, 3), (2, 4), (3, 0), (3, 4), (4, 1), (4, 4)]
    system_ids_chai = [(i, j) for i in range(5) for j in range(5)]
    system_ids_boltz = [(i, j) for i in range(5) for j in range(5)]
    system_ids_chai = sorted(list(set(system_ids_chai) - set(throw_chai)))
    system_ids_boltz = sorted(list(set(system_ids_boltz) - set(throw_boltz)))
    
    if rank<len(system_ids_chai):
        folder = 'chai'
        ids = system_ids_chai[rank]
    else:
        folder = 'boltz'
        rank_use = rank-len(system_ids_chai)
        if rank_use<len(system_ids_boltz):
            ids = system_ids_boltz[rank_use]
        else:
            ids = None
    if ids!=None:
        system_path = f'systems_{folder}/{ids[0]}/{ids[1]}'
        sim_length = 1 # ns
        timestep = 4 # fs
        n_steps = sim_length / timestep * 1000000
        
        try:
            run_md(
                path=system_path,
                 steps = n_steps)
        except Exception as e:
            print(e)
            pass

    else:
        print(f"Rank not used: {rank}")
        pass
