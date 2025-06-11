import numpy as np
import os
import pandas as pd
import json
from tqdm import tqdm
import antibody_design.utils.utils as utils
import argparse
import time
import numpy as np
from functools import wraps

def init_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

def mpi_timer(func):
    @wraps(func)
    def wrapper(filename, rank, comm, *args, **kwargs):
        start = time.time()
        result = func(filename, *args, **kwargs)
        end = time.time()
        duration = end - start

        output_line = f"{rank},{duration:.6f}\n"
        output_bytes = output_line.encode('utf-8')
        line_size = len(output_bytes)

        max_line_size = comm.allreduce(line_size, op=MPI.MAX)
        output_bytes = output_bytes.ljust(max_line_size, b' ')
        offset = rank * max_line_size

        file = MPI.File.Open(comm, filename,
                             MPI.MODE_CREATE | MPI.MODE_WRONLY)
        file.Write_at(offset, output_bytes)
        file.Close()

        comm.Barrier()
        if rank == 0:
            with open(filename, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write("rank,duration_seconds\n" + content)

        return result
    return wrapper


def _mpi_timer(filename, rank, comm):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            duration = end - start

            # Format output
            output_line = f"{rank},{duration:.6f}\n"
            output_bytes = output_line.encode('utf-8')
            line_size = len(output_bytes)

            # Ensure safe offsets by padding lines to same length
            max_line_size = comm.allreduce(line_size, op=MPI.MAX)
            output_bytes = output_bytes.ljust(max_line_size, b' ')
            offset = rank * max_line_size

            # Parallel write
            file = MPI.File.Open(comm, filename,
                                 MPI.MODE_CREATE | MPI.MODE_WRONLY)
            file.Write_at(offset, output_bytes)
            file.Close()

            # Optional header (serial, rank 0 only)
            comm.Barrier()
            if rank == 0:
                with open(filename, "r+") as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write("rank,duration_seconds\n" + content)

            return result
        return wrapper
    return decorator



