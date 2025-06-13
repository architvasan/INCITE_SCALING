import parsl
from parsl import python_app
from incite_bench.settings import AuroraSettings

@python_app
def run_mmpbsa(n_nodes: int=32,
               n_per_node: int = 200,
               last_frame: int = 50,
               system_str: str = 'inputs/small_target_system',
               AMBERHOME: str ='/flare/FoundEpidem/msinclair/envs/plinder/bin',
               benchmark: str = 'n_nodes_benchmarks'
               ):
    from mmpbsa import MMPBSA
    from pathlib import Path
    from random import random
    from time import perf_counter

    #AMBERHOME = '/flare/FoundEpidem/msinclair/envs/plinder/bin'
    #benchmark = 'n_nodes_benchmarks' # n_per_node; size; n_nodes
    system = Path(sytem_str)#'inputs/small_target_system')
    #n_per_node = 200
    #last_frame = 50 # 50 100 200
    
    top = Path(system) / 'system.prmtop'
    dcd = top.parent / 'prod.dcd'

    while True:
        rank = int(random() * 1000000)
        out = Path(benchmark) / system.name / f'{n_nodes}_nodes' / f'rank{rank}'
        if out.exists():
            continue
        else:
            break
    
    initial_time = perf_counter()
    mmpbsa = MMPBSA(top, dcd, selections=[':1-166', ':167-242'], out=out, last_frame=last_frame, amberhome=AMBERHOME)
    mmpbsa.run()
    final_time = perf_counter()
    
    timer = final_time - initial_time
    
    with open(f'time.txt', 'w') as fout:
        fout.write(f'{timer}')

def control_parsl(
        yaml_file: str = 'aurora.yaml',
        n_nodes: int = 32,
        n_per_node = 200,
        run_dir = '/flare/FoundEpidem/avasan/IDEAL/INCITE_Scaling/mmpbsa/',
        system_str = 'inputs/small_target_system',
        ):
    config = AuroraSettings.from_yaml(yaml_file).config_factory(run_dir)
    parsl.load(config)
    futures = []
    for i in range(n_nodes * n_per_node):
        futures.append(run_mmpbsa(n_nodes,
                                  n_per_node))
    
    _ = [x.result() for x in futures]


def main():
    parser = argparse.ArgumentParser(description="argparse mmpbsa")

    """
    yaml_file: str = 'aurora.yaml',
    n_nodes: int = 32,
    n_per_node = 200,
    run_dir = '/flare/FoundEpidem/msinclair/ideals/incite_benchmark'
    """

    # Add arguments
    parser.add_argument(
                "-y",
                "--yaml_file",
                type=str,
                help="Yaml File for parsl")

    parser.add_argument(
                "-R",
                "--run_dir",
                type=str,
                help="Directory to run parsl")
    
    parser.add_argument(
                "-N",
                "--n_nodes",
                type=int,
                help="Number of nodes")
    
    parser.add_argument("-n", "--n_per_node", type=int, help="Processes per node")
                                                                                          
    # Parse arguments
    args = parser.parse_args()

    control_parsl(args.yaml_file,
                  args.n_nodes,
                  args.n_per_node,
                  args.run_dir)
